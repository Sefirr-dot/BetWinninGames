"""
Both Teams To Score (BTTS).

Primary method (when Dixon-Coles lambda/mu are available):
  P(home scores) = 1 - e^{-lambda}   (exact Poisson CDF)
  P(away scores) = 1 - e^{-mu}
  P(BTTS)        = P(home scores) * P(away scores)

Fallback (no DC params): Bayesian blend of historical scoring/conceding rates.
"""

import math


def _scoring_rate(team_id: int, matches: list[dict], as_home: bool) -> float:
    """Fraction of matches where the team scored at least one goal."""
    relevant = []
    for m in matches:
        try:
            hg = m["score"]["fullTime"]["home"]
            ag = m["score"]["fullTime"]["away"]
            if hg is None or ag is None:
                continue
            if as_home and m["homeTeam"]["id"] == team_id:
                relevant.append(int(hg) > 0)
            elif not as_home and m["awayTeam"]["id"] == team_id:
                relevant.append(int(ag) > 0)
        except (KeyError, TypeError):
            continue

    if not relevant:
        return 0.65  # prior for scoring rate when no data
    return sum(relevant) / len(relevant)


def _conceding_rate(team_id: int, matches: list[dict], as_home: bool) -> float:
    """Fraction of matches where the team conceded at least one goal."""
    relevant = []
    for m in matches:
        try:
            hg = m["score"]["fullTime"]["home"]
            ag = m["score"]["fullTime"]["away"]
            if hg is None or ag is None:
                continue
            if as_home and m["homeTeam"]["id"] == team_id:
                relevant.append(int(ag) > 0)
            elif not as_home and m["awayTeam"]["id"] == team_id:
                relevant.append(int(hg) > 0)
        except (KeyError, TypeError):
            continue

    if not relevant:
        return 0.60
    return sum(relevant) / len(relevant)


def _league_btts_rate(matches: list[dict]) -> float:
    """Historical BTTS rate across all provided matches."""
    count = 0
    total = 0
    for m in matches:
        try:
            hg = m["score"]["fullTime"]["home"]
            ag = m["score"]["fullTime"]["away"]
            if hg is None or ag is None:
                continue
            total += 1
            if int(hg) > 0 and int(ag) > 0:
                count += 1
        except (KeyError, TypeError):
            continue
    return count / total if total > 0 else 0.50


def predict(
    home_id: int,
    away_id: int,
    all_matches: list[dict],
    lambda_: float | None = None,
    mu_: float | None = None,
) -> dict:
    """
    BTTS prediction.

    When lambda_ and mu_ (expected goals from Dixon-Coles) are provided, uses
    the mathematically exact Poisson formula.  Falls back to the historical
    Bayesian estimate otherwise.
    """
    prior = _league_btts_rate(all_matches)

    if lambda_ is not None and mu_ is not None and lambda_ > 0 and mu_ > 0:
        # Exact: P(team scores >= 1) = 1 - P(team scores 0) = 1 - e^{-lambda}
        p_home_scores = 1.0 - math.exp(-lambda_)
        p_away_scores = 1.0 - math.exp(-mu_)
        btts_prob = p_home_scores * p_away_scores
    else:
        # Fallback: Bayesian blend of historical rates
        h_scores  = _scoring_rate(home_id, all_matches, as_home=True)
        a_scores  = _scoring_rate(away_id, all_matches, as_home=False)
        h_concede = _conceding_rate(home_id, all_matches, as_home=True)
        a_concede = _conceding_rate(away_id, all_matches, as_home=False)

        p_home_scores = (h_scores + a_concede) / 2
        p_away_scores = (a_scores + h_concede) / 2
        likelihood = p_home_scores * p_away_scores
        btts_prob = 0.5 * prior + 0.5 * likelihood

    btts_prob = max(0.10, min(0.90, btts_prob))

    return {
        "btts_prob": btts_prob,
        "p_home_scores": p_home_scores,
        "p_away_scores": p_away_scores,
        "league_prior": prior,
    }
