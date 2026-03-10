"""
Dixon-Coles Poisson model.

Estimates attack (alpha) and defence (beta) parameters for each team via
maximum-likelihood estimation, with:
  - temporal decay (recent matches weighted more)
  - low-score correction (rho)
  - home advantage multiplier

Outputs a full scoreline probability matrix → 1X2, Over/Under, BTTS.
"""

import glob
import hashlib
import json
import math
import os
from typing import Optional
import numpy as np
from scipy.optimize import minimize
from datetime import datetime, date

from config import DC_XI, DC_XI_BY_LEAGUE, DC_RHO, HOME_ADVANTAGE

_DC_CACHE_DIR = "cache"
_DC_CACHE_PREFIX = "dc_params_"


def _params_cache_key(matches: list[dict], reference_date: date) -> str:
    """
    Fast fingerprint for cache invalidation.

    Hashes: total match count + reference_date + last 100 match dates.
    Cheap to compute (~1ms), sensitive to any new data fetched.
    """
    tail = sorted(m.get("utcDate", "")[:10] for m in matches[-100:])
    raw  = f"{len(matches)}|{reference_date}|{''.join(tail)}"
    return hashlib.md5(raw.encode()).hexdigest()[:16]


def _load_dc_cache(key: str) -> dict | None:
    path = os.path.join(_DC_CACHE_DIR, f"{_DC_CACHE_PREFIX}{key}.json")
    try:
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        # Restore integer team-ID keys (JSON serialises them as strings)
        for league_key, params in raw.items():
            if isinstance(params, dict):
                for d in ("alpha", "beta"):
                    if d in params:
                        params[d] = {int(k): v for k, v in params[d].items()}
        return raw
    except (FileNotFoundError, json.JSONDecodeError, KeyError, ValueError):
        return None


def _save_dc_cache(key: str, params: dict) -> None:
    os.makedirs(_DC_CACHE_DIR, exist_ok=True)
    # Remove stale cache files (keep only the latest)
    for old in glob.glob(os.path.join(_DC_CACHE_DIR, f"{_DC_CACHE_PREFIX}*.json")):
        try:
            os.remove(old)
        except OSError:
            pass
    path = os.path.join(_DC_CACHE_DIR, f"{_DC_CACHE_PREFIX}{key}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(params, f)


def _weight(match_date_str: str, reference_date: date, dc_xi: float = DC_XI) -> float:
    """Temporal decay weight for a match relative to the reference date."""
    try:
        d = datetime.strptime(match_date_str[:10], "%Y-%m-%d").date()
        days_ago = (reference_date - d).days
        if days_ago < 0:
            return 0.0
        return math.exp(-dc_xi * days_ago)
    except Exception:
        return 1.0


def _tau(x: int, y: int, lam: float, mu: float, rho: float) -> float:
    """Low-score correction factor."""
    if x == 0 and y == 0:
        return 1 - lam * mu * rho
    elif x == 0 and y == 1:
        return 1 + lam * rho
    elif x == 1 and y == 0:
        return 1 + mu * rho
    elif x == 1 and y == 1:
        return 1 - rho
    return 1.0


def _tau_sarmanov(x: int, y: int, lam: float, mu: float, theta: float) -> float:
    """Sarmanov bivariate correction for ALL scorelines.

    Factor: 1 + θ * f(x,λ) * g(y,μ)
    where f(x,λ) = (x−λ)/sqrt(λ)  and  g(y,μ) = (y−μ)/sqrt(μ)

    θ=0 → factor=1.0 (backward-compatible with models fitted without theta)
    """
    if theta == 0.0 or lam <= 0 or mu <= 0:
        return 1.0
    return 1.0 + theta * ((x - lam) / math.sqrt(lam)) * ((y - mu) / math.sqrt(mu))


def _poisson_pmf(k: int, lam: float) -> float:
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return math.exp(-lam) * (lam ** k) / math.factorial(k)


def fit(matches: list[dict], reference_date: Optional[date] = None, dc_xi: float = DC_XI) -> dict:
    """
    Fit Dixon-Coles model on a list of finished match dicts.

    Each match dict must have:
        homeTeam.id, awayTeam.id, score.fullTime.home, score.fullTime.away, utcDate

    Returns a dict with keys: alpha, beta, home_adv, rho
        - alpha[team_id] : attack strength
        - beta[team_id]  : defence weakness (higher = weaker)
    """
    if reference_date is None:
        reference_date = date.today()

    # Filter matches with valid scores
    valid = []
    for m in matches:
        try:
            hg = m["score"]["fullTime"]["home"]
            ag = m["score"]["fullTime"]["away"]
            if hg is None or ag is None:
                continue
            valid.append(m)
        except (KeyError, TypeError):
            continue

    if len(valid) < 20:
        return {}

    # Collect all teams
    teams = set()
    for m in valid:
        teams.add(m["homeTeam"]["id"])
        teams.add(m["awayTeam"]["id"])
    teams = sorted(teams)
    team_idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)

    weights = [_weight(m["utcDate"], reference_date, dc_xi) for m in valid]

    def neg_log_likelihood(params):
        alphas = np.exp(params[:n])   # attack (positive)
        betas  = np.exp(params[n:2*n])  # defence
        home_adv = math.exp(params[2*n])
        rho   = params[2*n + 1]
        theta = params[2*n + 2]

        nll = 0.0
        for m, w in zip(valid, weights):
            if w < 1e-9:
                continue
            hi = team_idx[m["homeTeam"]["id"]]
            ai = team_idx[m["awayTeam"]["id"]]
            # Use xG when available (enriched by understat_fetcher); fall back to actual goals
            hg = int(round(m["_xg_home"])) if "_xg_home" in m else int(m["score"]["fullTime"]["home"])
            ag = int(round(m["_xg_away"])) if "_xg_away" in m else int(m["score"]["fullTime"]["away"])

            lam = alphas[hi] * betas[ai] * home_adv
            mu  = alphas[ai] * betas[hi]

            tau_total = _tau(hg, ag, lam, mu, rho) * _tau_sarmanov(hg, ag, lam, mu, theta)
            if tau_total <= 0:
                tau_total = 1e-9

            p = (
                _poisson_pmf(hg, lam)
                * _poisson_pmf(ag, mu)
                * tau_total
            )
            if p <= 0:
                p = 1e-9
            nll -= w * math.log(p)

        # L2 regularisation on log-parameters.
        # Serves two purposes:
        #   1. Identifiability: alpha*c and beta/c give the same likelihood,
        #      the penalty anchors parameters near zero in log-space.
        #   2. Prevents overfitting for teams with few matches.
        nll += 0.001 * (float(np.sum(params[:n] ** 2)) +
                        float(np.sum(params[n:2 * n] ** 2)))
        return nll

    # Initial params: log(1) = 0 for all alphas/betas, log(home_adv), rho, theta
    x0 = np.zeros(2 * n + 3)
    x0[2 * n] = math.log(HOME_ADVANTAGE)
    x0[2 * n + 1] = DC_RHO
    # x0[2*n+2] = 0.0  (theta, already zero from np.zeros)

    # Bounds: rho in (-1, 1), theta in (-0.99, 0.99), rest unbounded
    bounds = [(None, None)] * (2 * n + 1) + [(-0.99, 0.99), (-0.99, 0.99)]

    result = minimize(
        neg_log_likelihood,
        x0,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 300, "ftol": 1e-8},
    )

    params = result.x
    alpha = {t: math.exp(params[team_idx[t]]) for t in teams}
    beta  = {t: math.exp(params[n + team_idx[t]]) for t in teams}
    home_adv = math.exp(params[2 * n])
    rho   = params[2 * n + 1]
    theta = params[2 * n + 2]

    return {
        "alpha": alpha,
        "beta": beta,
        "home_adv": home_adv,
        "rho": rho,
        "theta": theta,
        "teams": teams,
        "converged": result.success,
    }


def fit_per_league(matches: list[dict], reference_date: Optional[date] = None) -> dict:
    """
    Fit a separate Dixon-Coles model for each league found in the match list.

    Each match is expected to have a '_league_code' key (added by fetcher).
    Also fits a '_global' fallback model on all matches combined.

    Returns
    -------
    dict keyed by league code (e.g. "PL", "PD", ...) plus "_global".
    Each value is the params dict returned by fit(), or {} if fitting failed.
    """
    from collections import defaultdict

    if reference_date is None:
        reference_date = date.today()

    # Cache hit: if historical data hasn't changed since last run, skip fitting
    _cache_key = _params_cache_key(matches, reference_date)
    _cached    = _load_dc_cache(_cache_key)
    if _cached is not None:
        n_leagues = sum(1 for k in _cached if k != "_global" and _cached[k])
        print(f"    DC [cache]: {n_leagues} ligas cargadas sin re-ajuste "
              f"(datos sin cambios)")
        return _cached

    by_league: dict[str, list] = defaultdict(list)
    for m in matches:
        code = m.get("_league_code")
        if code:
            by_league[code].append(m)

    result: dict[str, dict] = {}
    for code, league_matches in by_league.items():
        xi = DC_XI_BY_LEAGUE.get(code, DC_XI)
        params = fit(league_matches, reference_date, dc_xi=xi)
        result[code] = params
        if params:
            status = f"OK ({len(league_matches)} partidos, xi={xi})"
        else:
            status = "OMITIDO (pocos datos)"
        print(f"    DC [{code}]: {status}")

    # Global fallback for teams not found in any per-league model
    result["_global"] = fit(matches, reference_date)

    _save_dc_cache(_cache_key, result)
    return result


def predict(
    home_id: int,
    away_id: int,
    params: dict,
    max_goals: int = 8,
    home_fatigue: float = 1.0,
    away_fatigue: float = 1.0,
) -> dict:
    """
    Generate full probability distributions for a match.

    Returns:
        lambda_  : expected home goals
        mu_      : expected away goals
        prob_matrix : 2-D array [home_goals][away_goals]
        prob_home, prob_draw, prob_away
        over25, over35
        btts
        most_likely_score : (h, a)
        most_likely_score_prob
    """
    if not params or home_id not in params["alpha"] or away_id not in params["alpha"]:
        return {}

    lam   = params["alpha"][home_id] * params["beta"][away_id] * params["home_adv"] * home_fatigue
    mu    = params["alpha"][away_id] * params["beta"][home_id] * away_fatigue
    rho   = params["rho"]
    theta = params.get("theta", 0.0)   # 0.0 for params fitted without Sarmanov

    size = max_goals + 1
    matrix = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            t = _tau(i, j, lam, mu, rho) * _tau_sarmanov(i, j, lam, mu, theta)
            matrix[i][j] = max(0.0, _poisson_pmf(i, lam) * _poisson_pmf(j, mu) * t)

    # Normalise (rho correction can cause slight deviation from 1)
    total = matrix.sum()
    if total > 0:
        matrix /= total

    prob_home = float(np.sum(np.tril(matrix, -1)))
    prob_draw = float(np.trace(matrix))
    prob_away = float(np.sum(np.triu(matrix, 1)))

    over25 = float(sum(
        matrix[i][j] for i in range(size) for j in range(size) if i + j > 2
    ))
    over35 = float(sum(
        matrix[i][j] for i in range(size) for j in range(size) if i + j > 3
    ))

    btts = float(sum(
        matrix[i][j] for i in range(size) for j in range(size) if i > 0 and j > 0
    ))

    # Most likely score
    max_idx = np.unravel_index(np.argmax(matrix), matrix.shape)
    most_likely = (int(max_idx[0]), int(max_idx[1]))

    return {
        "lambda_": lam,
        "mu_": mu,
        "prob_matrix": matrix,
        "prob_home": prob_home,
        "prob_draw": prob_draw,
        "prob_away": prob_away,
        "over25": over25,
        "over35": over35,
        "btts": btts,
        "most_likely_score": most_likely,
        "most_likely_score_prob": float(matrix[most_likely[0]][most_likely[1]]),
    }
