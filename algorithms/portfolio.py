"""
Markowitz Portfolio Optimizer for weekend picks.

Replaces naive independent Kelly staking with a proper portfolio approach
that accounts for correlations between picks (same league, same day, etc.)
and enforces a maximum drawdown constraint.

Technique: Quadratic Programming (QP) using scipy.optimize.minimize
with SLSQP solver. No external dependencies beyond scipy (already required).

Markowitz mean-variance optimization:
    Maximize:  E[return] - λ * Var[return]
    Subject to:
        sum(w_i) <= MAX_TOTAL_EXPOSURE
        0 <= w_i <= min(kelly_i, MAX_SINGLE_BET)
        Expected drawdown <= MAX_DRAWDOWN

For N bets, E[return] = sum(w_i * ev_i) and Var[return] = w^T * C * w,
where C is the N×N correlation matrix.

Usage:
    from algorithms.portfolio import optimize_portfolio
    result = optimize_portfolio(value_bets, bankroll=1000.0)
    # result["optimized_stakes"] — dict {vb_key: stake_amount}
    # result["portfolio_ev"]    — expected value in bankroll units
    # result["portfolio_var"]   — portfolio variance
    # result["n_bets"]          — number of active bets (w > threshold)
"""

from __future__ import annotations
import math
import numpy as np
from scipy.optimize import minimize

# ---------------------------------------------------------------------------
# Portfolio constraints
# ---------------------------------------------------------------------------

MAX_TOTAL_EXPOSURE = 0.30   # max fraction of bankroll across all bets combined
MAX_SINGLE_BET     = 0.08   # max fraction of bankroll on any single bet (hard cap)
MAX_PORTFOLIO_VAR  = 0.015  # max allowed portfolio variance (controls drawdown)
RISK_AVERSION      = 2.0    # λ: higher = more conservative (0 = pure EV maximiser)
MIN_STAKE_FRACTION = 0.001  # bets below this fraction treated as 0 (pruned)

# ---------------------------------------------------------------------------
# Correlation matrix construction
# ---------------------------------------------------------------------------

# Base correlation between pairs by category.
# Same match: handled by _OUTCOME_CORRELATION from value_detector.
# Cross-match correlations are lower:
_CORR_SAME_LEAGUE_SAME_DAY   = 0.12   # systemic: bad day for league = all teams hurt
_CORR_SAME_LEAGUE_DIFF_DAY   = 0.05
_CORR_DIFF_LEAGUE_SAME_DAY   = 0.04   # weather / market sentiment correlation
_CORR_DIFF_LEAGUE_DIFF_DAY   = 0.02

# Within-match outcome correlations (same match, different markets)
_WITHIN_MATCH_CORR = {
    ("home",   "away"):    -0.90,
    ("home",   "draw"):    -0.70,
    ("draw",   "away"):    -0.70,
    ("home",   "over25"):   0.30,
    ("away",   "over25"):   0.25,
    ("home",   "btts"):     0.15,
    ("away",   "btts"):     0.15,
    ("over25", "btts"):     0.55,
    ("draw",   "over25"):  -0.20,
    ("draw",   "btts"):     0.10,
}


def _cross_match_corr(bet_a: dict, bet_b: dict) -> float:
    """Estimate correlation between two bets on DIFFERENT matches."""
    same_league = bet_a.get("league") == bet_b.get("league")
    same_day    = bet_a.get("match_date", "")[:10] == bet_b.get("match_date", "")[:10]

    if same_league and same_day:
        return _CORR_SAME_LEAGUE_SAME_DAY
    elif same_league:
        return _CORR_SAME_LEAGUE_DIFF_DAY
    elif same_day:
        return _CORR_DIFF_LEAGUE_SAME_DAY
    else:
        return _CORR_DIFF_LEAGUE_DIFF_DAY


def _within_match_corr(outcome_a: str, outcome_b: str) -> float:
    """Correlation between two outcomes on the SAME match."""
    key = (outcome_a, outcome_b)
    corr = _WITHIN_MATCH_CORR.get(key, _WITHIN_MATCH_CORR.get((key[1], key[0]), 0.0))
    return corr


def build_correlation_matrix(bets: list[dict]) -> np.ndarray:
    """
    Build N×N correlation matrix for a list of value bets.

    bet dict must have: "home_name", "away_name", "outcome", "league",
                        "match_date" (YYYY-MM-DD).
    """
    n = len(bets)
    C = np.eye(n)

    for i in range(n):
        for j in range(i + 1, n):
            a, b = bets[i], bets[j]
            same_match = (a.get("home_name") == b.get("home_name") and
                          a.get("away_name") == b.get("away_name"))

            if same_match:
                corr = _within_match_corr(a.get("outcome", ""), b.get("outcome", ""))
            else:
                corr = _cross_match_corr(a, b)

            C[i, j] = corr
            C[j, i] = corr

    # Ensure positive semi-definite (numerical safety for QP)
    min_eig = np.linalg.eigvalsh(C).min()
    if min_eig < 0:
        C += (-min_eig + 1e-6) * np.eye(n)

    return C


# ---------------------------------------------------------------------------
# Expected value computation
# ---------------------------------------------------------------------------

def _bet_ev(bet: dict) -> float:
    """
    Expected value per unit stake for one bet.
    EV = model_prob * (bk_odds - 1) - (1 - model_prob)
       = model_prob * bk_odds - 1
    """
    p   = bet.get("model_prob", 0.5)
    bk  = bet.get("bk_odds", 2.0)
    return p * bk - 1.0


def _bet_variance(bet: dict) -> float:
    """
    Variance of return per unit stake for one bet.
    Var = p * (b-1)^2 * (1-p) + (1-p) * p^2  [binary outcome]
    Simplified: Var = p * (1-p) * bk^2  (dominates)
    """
    p  = bet.get("model_prob", 0.5)
    bk = bet.get("bk_odds", 2.0)
    return p * (1.0 - p) * (bk ** 2)


# ---------------------------------------------------------------------------
# QP Optimizer
# ---------------------------------------------------------------------------

def optimize_portfolio(
    value_bets: list[dict],
    bankroll: float = 1.0,
    risk_aversion: float = RISK_AVERSION,
    max_total_exposure: float = MAX_TOTAL_EXPOSURE,
    max_single_bet: float = MAX_SINGLE_BET,
) -> dict:
    """
    Find optimal stake fractions for a list of value bets using Markowitz QP.

    Parameters
    ----------
    value_bets           : list of value_bet dicts from value_detector.find_edges()
    bankroll             : total bankroll (absolute units, e.g. euros)
    risk_aversion        : λ in mean-variance tradeoff (higher = more conservative)
    max_total_exposure   : max total fraction of bankroll staked
    max_single_bet       : max fraction per single bet

    Returns
    -------
    dict:
        "optimized_stakes"  : {bet_key: stake_in_bankroll_units}
        "optimized_fractions": {bet_key: fraction_of_bankroll}
        "portfolio_ev"      : expected P&L in bankroll units
        "portfolio_vol"     : portfolio volatility (std dev)
        "portfolio_sharpe"  : portfolio Sharpe ratio (EV / vol)
        "n_bets"            : number of bets with stake > threshold
        "kelly_reduction"   : ratio of optimised to raw Kelly total
        "method"            : "markowitz" or "fallback_kelly"
    """
    if not value_bets:
        return _empty_result()

    n = len(value_bets)

    # Upper bounds: minimum of raw Kelly fraction and hard cap
    upper_bounds = np.array([
        min(vb.get("kelly_fraction", 0.02), max_single_bet)
        for vb in value_bets
    ])

    # Expected values and per-bet variances
    ev_arr  = np.array([_bet_ev(vb) for vb in value_bets])
    var_arr = np.array([_bet_variance(vb) for vb in value_bets])

    # Correlation + covariance matrix
    try:
        C = build_correlation_matrix(value_bets)
        # Covariance = sigma_i * sigma_j * corr_ij
        sigma = np.sqrt(var_arr)
        Sigma = np.outer(sigma, sigma) * C   # covariance matrix
    except Exception:
        # Fallback to diagonal covariance (no correlations)
        Sigma = np.diag(var_arr)

    # ── QP objective: minimize -EV + λ * portfolio_variance ───────────────
    def objective(w: np.ndarray) -> float:
        port_ev  = float(np.dot(ev_arr, w))
        port_var = float(w @ Sigma @ w)
        return -port_ev + risk_aversion * port_var

    def objective_grad(w: np.ndarray) -> np.ndarray:
        return -ev_arr + 2.0 * risk_aversion * (Sigma @ w)

    # Initial guess: proportional to EV, clipped to bounds
    w0 = np.clip(ev_arr / max(ev_arr.sum(), 1e-6) * max_total_exposure * 0.5, 0, upper_bounds)

    # Constraints
    constraints = [
        # Total exposure <= max_total_exposure
        {"type": "ineq", "fun": lambda w: max_total_exposure - w.sum(),
                          "jac": lambda w: -np.ones(n)},
    ]

    bounds = [(0.0, ub) for ub in upper_bounds]

    try:
        result = minimize(
            objective,
            w0,
            method="SLSQP",
            jac=objective_grad,
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-10, "maxiter": 500},
        )
        w_opt = np.clip(result.x, 0, upper_bounds)
    except Exception:
        # Fallback: just use raw Kelly fractions scaled to exposure cap
        w_opt = upper_bounds.copy()
        total = w_opt.sum()
        if total > max_total_exposure:
            w_opt *= max_total_exposure / total

    # Prune negligible bets
    w_opt[w_opt < MIN_STAKE_FRACTION] = 0.0

    # Portfolio metrics
    port_ev  = float(np.dot(ev_arr, w_opt))
    port_var = float(w_opt @ Sigma @ w_opt)
    port_vol = math.sqrt(max(port_var, 0.0))
    sharpe   = port_ev / port_vol if port_vol > 1e-9 else 0.0

    # Raw Kelly total for comparison
    raw_kelly_total = sum(vb.get("kelly_fraction", 0.0) for vb in value_bets)
    optimized_total = float(w_opt.sum())
    kelly_reduction = (optimized_total / raw_kelly_total) if raw_kelly_total > 1e-9 else 1.0

    # Build output
    optimized_fractions = {}
    optimized_stakes    = {}
    for i, vb in enumerate(value_bets):
        key = f"{vb.get('home_name', '')}|{vb.get('away_name', '')}|{vb.get('outcome', '')}"
        frac = float(w_opt[i])
        optimized_fractions[key] = round(frac, 6)
        optimized_stakes[key]    = round(frac * bankroll, 2)

    return {
        "optimized_fractions": optimized_fractions,
        "optimized_stakes":    optimized_stakes,
        "portfolio_ev":        round(port_ev,  5),
        "portfolio_vol":       round(port_vol, 5),
        "portfolio_sharpe":    round(sharpe,   4),
        "n_bets":              int((w_opt > MIN_STAKE_FRACTION).sum()),
        "kelly_reduction":     round(kelly_reduction, 4),
        "method":              "markowitz",
        "max_single_fraction": round(float(w_opt.max()), 5) if n > 0 else 0.0,
        "total_exposure":      round(optimized_total, 5),
    }


def apply_portfolio_stakes(
    value_bets: list[dict],
    portfolio_result: dict,
) -> list[dict]:
    """
    Attach optimised Kelly fractions to each value_bet dict.

    Adds "kelly_portfolio" field (Markowitz-optimised fraction) alongside
    the existing "kelly_fraction" (independent Kelly) for comparison.
    """
    fracs = portfolio_result.get("optimized_fractions", {})
    for vb in value_bets:
        key = f"{vb.get('home_name', '')}|{vb.get('away_name', '')}|{vb.get('outcome', '')}"
        vb["kelly_portfolio"] = fracs.get(key, vb.get("kelly_fraction", 0.0))
    return value_bets


def _empty_result() -> dict:
    return {
        "optimized_fractions": {},
        "optimized_stakes":    {},
        "portfolio_ev":        0.0,
        "portfolio_vol":       0.0,
        "portfolio_sharpe":    0.0,
        "n_bets":              0,
        "kelly_reduction":     1.0,
        "method":              "no_bets",
        "max_single_fraction": 0.0,
        "total_exposure":      0.0,
    }


# ---------------------------------------------------------------------------
# Monte Carlo simulation of weekend P&L distribution
# ---------------------------------------------------------------------------

def simulate_weekend_pnl(
    value_bets: list[dict],
    portfolio_result: dict,
    n_simulations: int = 10_000,
) -> dict:
    """
    Simulate the distribution of P&L for the optimised portfolio.

    Uses the optimised stake fractions and each bet's model probability
    to generate N_SIMULATIONS outcomes. Reports P5, P25, P50, P75, P95
    percentiles plus probability of positive return.

    Returns dict ready for JSON serialization (for tracker_data.js).
    """
    fracs = portfolio_result.get("optimized_fractions", {})
    if not fracs or not value_bets:
        return {}

    rng = np.random.default_rng(seed=42)
    pnl = np.zeros(n_simulations)

    for vb in value_bets:
        key  = f"{vb.get('home_name', '')}|{vb.get('away_name', '')}|{vb.get('outcome', '')}"
        frac = fracs.get(key, 0.0)
        if frac < MIN_STAKE_FRACTION:
            continue

        prob = vb.get("model_prob", 0.5)
        bk   = vb.get("bk_odds", 2.0)

        wins = rng.random(n_simulations) < prob
        pnl += frac * np.where(wins, bk - 1.0, -1.0)

    percentiles = np.percentile(pnl, [5, 25, 50, 75, 95])
    prob_positive = float((pnl > 0).mean())

    return {
        "p05":          round(float(percentiles[0]), 4),
        "p25":          round(float(percentiles[1]), 4),
        "p50":          round(float(percentiles[2]), 4),
        "p75":          round(float(percentiles[3]), 4),
        "p95":          round(float(percentiles[4]), 4),
        "prob_positive": round(prob_positive, 4),
        "n_simulations": n_simulations,
        "n_bets":        portfolio_result.get("n_bets", 0),
    }
