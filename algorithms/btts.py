"""
Both Teams To Score (BTTS).

Primary method (when Dixon-Coles lambda/mu are available):
  P(home scores) = 1 - e^{-lambda}   (exact Poisson CDF)
  P(away scores) = 1 - e^{-mu}
  P(BTTS)        = P(home scores) * P(away scores)

Fallback (no DC params): Bayesian blend of venue-specific scoring/conceding
rates with automatic fallback to global rates when venue sample is too small.
"""

import math

from config import BTTS_PRIOR_BLEND, BTTS_RATE_BY_LEAGUE

_MIN_VENUE_MATCHES = 5   # minimum venue-specific matches before using venue rate


def _scoring_rate(
    team_id: int,
    matches: list[dict],
    as_home: bool | None,
    _depth: int = 0,
) -> float:
    """
    Fraction of matches where the team scored at least one goal.

    as_home=True  → only home matches for this team
    as_home=False → only away matches for this team
    as_home=None  → all matches (global fallback)

    When a venue-specific call has fewer than _MIN_VENUE_MATCHES samples,
    falls back to the global rate (as_home=None) instead of a hardcoded prior.
    """
    relevant = []
    for m in matches:
        try:
            hg = m["score"]["fullTime"]["home"]
            ag = m["score"]["fullTime"]["away"]
            if hg is None or ag is None:
                continue
            if as_home is True and m["homeTeam"]["id"] == team_id:
                relevant.append(int(hg) > 0)
            elif as_home is False and m["awayTeam"]["id"] == team_id:
                relevant.append(int(ag) > 0)
            elif as_home is None:
                if m["homeTeam"]["id"] == team_id:
                    relevant.append(int(hg) > 0)
                elif m["awayTeam"]["id"] == team_id:
                    relevant.append(int(ag) > 0)
        except (KeyError, TypeError):
            continue

    # Venue-specific path: fall back to global if not enough samples
    if as_home is not None and len(relevant) < _MIN_VENUE_MATCHES and _depth == 0:
        return _scoring_rate(team_id, matches, as_home=None, _depth=1)

    if not relevant:
        return 0.65   # last-resort prior (team with no data at all)
    return sum(relevant) / len(relevant)


def _conceding_rate(
    team_id: int,
    matches: list[dict],
    as_home: bool | None,
    _depth: int = 0,
) -> float:
    """
    Fraction of matches where the team conceded at least one goal.

    Same venue-specific logic and fallback as _scoring_rate.
    """
    relevant = []
    for m in matches:
        try:
            hg = m["score"]["fullTime"]["home"]
            ag = m["score"]["fullTime"]["away"]
            if hg is None or ag is None:
                continue
            if as_home is True and m["homeTeam"]["id"] == team_id:
                relevant.append(int(ag) > 0)
            elif as_home is False and m["awayTeam"]["id"] == team_id:
                relevant.append(int(hg) > 0)
            elif as_home is None:
                if m["homeTeam"]["id"] == team_id:
                    relevant.append(int(ag) > 0)
                elif m["awayTeam"]["id"] == team_id:
                    relevant.append(int(hg) > 0)
        except (KeyError, TypeError):
            continue

    if as_home is not None and len(relevant) < _MIN_VENUE_MATCHES and _depth == 0:
        return _conceding_rate(team_id, matches, as_home=None, _depth=1)

    if not relevant:
        return 0.60   # last-resort prior
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
    league_code: str | None = None,
) -> dict:
    """
    BTTS prediction.

    When lambda_ and mu_ (expected goals from Dixon-Coles) are provided, uses
    the mathematically exact Poisson formula.  Falls back to the historical
    Bayesian estimate with venue-specific rates otherwise.

    In both paths, the result is blended with the per-league historical BTTS
    rate (BTTS_PRIOR_BLEND) to correct the systematic underestimation at low
    probabilities identified by the backtest calibration.
    """
    # Per-league prior (more accurate than the match-corpus rate)
    league_prior = BTTS_RATE_BY_LEAGUE.get(league_code) if league_code else None
    if league_prior is None:
        league_prior = _league_btts_rate(all_matches)

    if lambda_ is not None and mu_ is not None and lambda_ > 0 and mu_ > 0:
        p_home_scores = 1.0 - math.exp(-lambda_)
        p_away_scores = 1.0 - math.exp(-mu_)
        btts_poisson  = p_home_scores * p_away_scores
        # Blend with league prior to raise the floor for low-prob games.
        # Backtest: 0.3-0.4 bucket actual=46.7% vs predicted=36.3% (+10.5% gap).
        btts_prob = (1 - BTTS_PRIOR_BLEND) * btts_poisson + BTTS_PRIOR_BLEND * league_prior
    else:
        h_scores  = _scoring_rate(home_id,  all_matches, as_home=True)
        a_scores  = _scoring_rate(away_id,  all_matches, as_home=False)
        h_concede = _conceding_rate(home_id, all_matches, as_home=True)
        a_concede = _conceding_rate(away_id, all_matches, as_home=False)

        p_home_scores = (h_scores + a_concede) / 2
        p_away_scores = (a_scores + h_concede) / 2
        likelihood = p_home_scores * p_away_scores
        btts_prob  = 0.5 * league_prior + 0.5 * likelihood

    btts_prob = max(0.10, min(0.90, btts_prob))

    return {
        "btts_prob":      btts_prob,
        "p_home_scores":  p_home_scores,
        "p_away_scores":  p_away_scores,
        "league_prior":   league_prior,
    }
