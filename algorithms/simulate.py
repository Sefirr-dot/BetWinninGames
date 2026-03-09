"""
Monte Carlo match simulator.

Replaces Poisson-exact calculations for secondary markets with vectorized
NumPy simulation. Advantages over the matrix approach:
  - No goal cap (matrix caps at max_goals=8)
  - Trivially extensible to any market (Over 3.5, Asian HCap, BTTS+O25, ...)
  - Applies Dixon-Coles rho correction via importance weighting
  - ~2ms for 50k simulations (vectorized numpy)

Usage
-----
    from algorithms.simulate import simulate
    mc = simulate(lambda_=1.4, mu_=1.1, rho=0.05)
    print(mc["over35"], mc["btts_and_over25"], mc["ah_home_minus1_win"])
"""

import numpy as np

_N_SIMULATIONS = 50_000
_RNG = np.random.default_rng(seed=42)


def simulate(
    lambda_: float,
    mu_: float,
    rho: float = 0.05,
    n: int = _N_SIMULATIONS,
) -> dict:
    """
    Simulate n matches using Poisson(lambda_) and Poisson(mu_).
    Applies the Dixon-Coles low-score correction (rho) as importance weights.

    Parameters
    ----------
    lambda_ : expected home goals (fatigue-adjusted, from DC)
    mu_     : expected away goals
    rho     : Dixon-Coles low-score correction parameter
    n       : number of simulations

    Returns
    -------
    dict with probabilities for all markets (0.0–1.0 range)
    """
    if lambda_ <= 0 or mu_ <= 0:
        return _empty()

    hg = _RNG.poisson(max(lambda_, 1e-6), n).astype(np.int32)
    ag = _RNG.poisson(max(mu_,      1e-6), n).astype(np.int32)
    diff  = hg - ag
    total = hg + ag

    # ── Dixon-Coles importance weights for low scores ─────────────────────
    w = np.ones(n, dtype=np.float64)
    w[(hg == 0) & (ag == 0)] *= max(0.0, 1.0 - lambda_ * mu_ * rho)
    w[(hg == 1) & (ag == 0)] *= max(0.0, 1.0 + mu_ * rho)
    w[(hg == 0) & (ag == 1)] *= max(0.0, 1.0 + lambda_ * rho)
    w[(hg == 1) & (ag == 1)] *= max(0.0, 1.0 - rho)
    w_sum = w.sum()
    if w_sum < 1e-9:
        w = np.full(n, 1.0 / n)
    else:
        w /= w_sum

    def p(cond: np.ndarray) -> float:
        return float(np.dot(w, cond.astype(np.float64)))

    # ── Goal markets ──────────────────────────────────────────────────────
    over05  = p(total > 0)
    over15  = p(total > 1)
    over25  = p(total > 2)
    over35  = p(total > 3)
    over45  = p(total > 4)
    over55  = p(total > 5)

    # ── BTTS & combinations ───────────────────────────────────────────────
    btts_mask = (hg > 0) & (ag > 0)
    btts           = p(btts_mask)
    btts_and_over25 = p(btts_mask & (total > 2))
    btts_and_over15 = p(btts_mask & (total > 1))

    # ── 1X2 ──────────────────────────────────────────────────────────────
    prob_home = p(diff > 0)
    prob_draw = p(diff == 0)
    prob_away = p(diff < 0)

    # ── Asian Handicap — home team perspective ────────────────────────────
    # AH -0.5 home = home must win (same as 1X2 home win)
    # AH -1   home = full win if home wins by 2+, push if wins by exactly 1
    # AH -1.5 home = full win only if home wins by 2+
    # AH +0.5 away = away must not lose
    # AH +1   away = full win if away wins or draws, push if home wins by 1
    ah_home_m05_win   = prob_home                        # home wins
    ah_home_m1_win    = p(diff > 1)                      # home wins by 2+
    ah_home_m1_push   = p(diff == 1)                     # home wins by exactly 1 → push
    ah_home_m1_loss   = p(diff <= 0)
    ah_home_m15_win   = p(diff > 1)                      # home wins by 2+ (no push)
    ah_away_p05_win   = p(diff <= 0)                     # away wins or draws
    ah_away_p1_win    = p(diff < 1)                      # away wins or home wins by exactly 1 → push
    ah_away_p1_push   = p(diff == 1)
    ah_away_p15_win   = p(diff < 2)                      # away doesn't lose by 2+

    return {
        # Goal total markets
        "over05":  round(over05,  4),
        "over15":  round(over15,  4),
        "over25":  round(over25,  4),
        "over35":  round(over35,  4),
        "over45":  round(over45,  4),
        "over55":  round(over55,  4),
        # BTTS
        "btts":            round(btts,            4),
        "btts_and_over15": round(btts_and_over15, 4),
        "btts_and_over25": round(btts_and_over25, 4),
        # 1X2
        "prob_home": round(prob_home, 4),
        "prob_draw": round(prob_draw, 4),
        "prob_away": round(prob_away, 4),
        # Asian Handicap
        "ah_home_m05_win":  round(ah_home_m05_win,  4),
        "ah_home_m1_win":   round(ah_home_m1_win,   4),
        "ah_home_m1_push":  round(ah_home_m1_push,  4),
        "ah_home_m1_loss":  round(ah_home_m1_loss,  4),
        "ah_home_m15_win":  round(ah_home_m15_win,  4),
        "ah_away_p05_win":  round(ah_away_p05_win,  4),
        "ah_away_p1_win":   round(ah_away_p1_win,   4),
        "ah_away_p1_push":  round(ah_away_p1_push,  4),
        "ah_away_p15_win":  round(ah_away_p15_win,  4),
    }


def _empty() -> dict:
    return {k: 0.5 for k in (
        "over05", "over15", "over25", "over35", "over45", "over55",
        "btts", "btts_and_over15", "btts_and_over25",
        "prob_home", "prob_draw", "prob_away",
        "ah_home_m05_win", "ah_home_m1_win", "ah_home_m1_push",
        "ah_home_m1_loss", "ah_home_m15_win",
        "ah_away_p05_win", "ah_away_p1_win", "ah_away_p1_push", "ah_away_p15_win",
    )}
