"""
Elo rating system.

All teams start at ELO_INITIAL. Ratings are updated chronologically through
all historical matches. Home team gets ELO_HOME_BONUS advantage during
probability calculation (but NOT stored in the rating).
"""

import math
from config import ELO_INITIAL, ELO_K, ELO_HOME_BONUS, ELO_HOME_BONUS_BY_LEAGUE, ELO_GOAL_DIFF_EXP, ELO_SEASON_REGRESSION


def _season_id(match: dict):
    """Extract season identifier from a match dict (API season.id or year from utcDate)."""
    season = match.get("season")
    if isinstance(season, dict):
        return season.get("id")
    # Fallback: use the year portion of utcDate
    try:
        return match["utcDate"][:4]
    except (KeyError, TypeError):
        return None


def build_ratings(matches: list[dict]) -> dict[int, float]:
    """
    Process matches in chronological order and return final Elo ratings.

    Improvements over plain Elo:
    - Goal-difference multiplier: larger wins update ratings more.
      K_effective = K * (1 + goal_diff) ** ELO_GOAL_DIFF_EXP
    - Season regression: at every new season, ratings are pulled back toward
      ELO_INITIAL by factor ELO_SEASON_REGRESSION, reflecting squad changes.
    """
    ratings: dict[int, float] = {}
    current_season = None

    sorted_matches = sorted(matches, key=lambda m: m.get("utcDate", ""))

    for m in sorted_matches:
        try:
            hg = m["score"]["fullTime"]["home"]
            ag = m["score"]["fullTime"]["away"]
            if hg is None or ag is None:
                continue
            hg, ag = int(hg), int(ag)
        except (KeyError, TypeError):
            continue

        # --- Season regression ---
        season = _season_id(m)
        if season is not None and season != current_season:
            if current_season is not None:
                # New season detected: regress all known ratings toward the mean
                for team_id in ratings:
                    ratings[team_id] = (
                        ELO_INITIAL
                        + (ratings[team_id] - ELO_INITIAL) * ELO_SEASON_REGRESSION
                    )
            current_season = season

        h_id = m["homeTeam"]["id"]
        a_id = m["awayTeam"]["id"]

        rh = ratings.get(h_id, ELO_INITIAL)
        ra = ratings.get(a_id, ELO_INITIAL)

        # Expected score (with league-specific home bonus)
        league = m.get("_league_code", "")
        bonus = ELO_HOME_BONUS_BY_LEAGUE.get(league, ELO_HOME_BONUS)
        eh = 1 / (1 + 10 ** ((ra - (rh + bonus)) / 400))
        ea = 1 - eh

        # Actual score
        if hg > ag:
            sh, sa = 1.0, 0.0
        elif hg < ag:
            sh, sa = 0.0, 1.0
        else:
            sh, sa = 0.5, 0.5

        # Goal-difference multiplier: a 4-0 updates more than a 1-0
        goal_diff = abs(hg - ag)
        k_eff = ELO_K * (1 + goal_diff) ** ELO_GOAL_DIFF_EXP

        ratings[h_id] = rh + k_eff * (sh - eh)
        ratings[a_id] = ra + k_eff * (sa - ea)

    return ratings


def predict(
    home_id: int,
    away_id: int,
    ratings: dict[int, float],
    league: str = "",
    home_ratings: dict[int, float] | None = None,
    away_ratings: dict[int, float] | None = None,
) -> dict:
    """
    Return 1X2 probabilities from Elo ratings.

    Home advantage is applied via league-specific ELO_HOME_BONUS_BY_LEAGUE.

    Parameters
    ----------
    ratings       : combined Elo ratings dict (fallback when split not provided).
    home_ratings  : venue-specific home ratings from build_split_ratings().
                    When provided, used for the home team instead of ratings.
    away_ratings  : venue-specific away ratings from build_split_ratings().
                    When provided, used for the away team instead of ratings.
    """
    rh = (home_ratings.get(home_id) if home_ratings else None)
    if rh is None:
        rh = ratings.get(home_id, ELO_INITIAL)
    ra = (away_ratings.get(away_id) if away_ratings else None)
    if ra is None:
        ra = ratings.get(away_id, ELO_INITIAL)

    # Win probability for home (with league-specific bonus)
    bonus = ELO_HOME_BONUS_BY_LEAGUE.get(league, ELO_HOME_BONUS)
    p_home_win = 1 / (1 + 10 ** ((ra - (rh + bonus)) / 400))

    # Draw probability: Elo doesn't naturally give draws.
    # Use a calibrated empirical approximation.
    draw_base = 0.25
    diff = abs(p_home_win - 0.5)
    prob_draw = max(0.05, draw_base - 0.3 * diff)

    prob_home = p_home_win * (1 - prob_draw)
    prob_away = (1 - p_home_win) * (1 - prob_draw)

    # Normalise to sum to 1
    total = prob_home + prob_draw + prob_away
    return {
        "prob_home": prob_home / total,
        "prob_draw": prob_draw / total,
        "prob_away": prob_away / total,
        "rating_home": rh,
        "rating_away": ra,
    }


def build_split_ratings(
    matches: list[dict],
) -> tuple[dict[int, float], dict[int, float]]:
    """
    Build separate home and away Elo ratings by processing each match only
    for the role in which each team played.

    - Home rating  : updated only when the team plays at home.
    - Away rating  : updated only when the team plays away.
    - Season regression applied to both independently.

    This captures venue-specific form that a single combined rating averages
    away (e.g. a team that dominates at home but struggles away).

    Returns
    -------
    (home_ratings, away_ratings) — each is {team_id: elo_rating}.
    Teams with no home games default to ELO_INITIAL when looked up.
    """
    home_ratings: dict[int, float] = {}
    away_ratings: dict[int, float] = {}
    current_season = None

    sorted_matches = sorted(matches, key=lambda m: m.get("utcDate", ""))

    for m in sorted_matches:
        try:
            hg = m["score"]["fullTime"]["home"]
            ag = m["score"]["fullTime"]["away"]
            if hg is None or ag is None:
                continue
            hg, ag = int(hg), int(ag)
        except (KeyError, TypeError):
            continue

        # Season regression applied to both dicts independently
        season = _season_id(m)
        if season is not None and season != current_season:
            if current_season is not None:
                for tid in home_ratings:
                    home_ratings[tid] = (
                        ELO_INITIAL
                        + (home_ratings[tid] - ELO_INITIAL) * ELO_SEASON_REGRESSION
                    )
                for tid in away_ratings:
                    away_ratings[tid] = (
                        ELO_INITIAL
                        + (away_ratings[tid] - ELO_INITIAL) * ELO_SEASON_REGRESSION
                    )
            current_season = season

        h_id = m["homeTeam"]["id"]
        a_id = m["awayTeam"]["id"]

        rh = home_ratings.get(h_id, ELO_INITIAL)
        ra = away_ratings.get(a_id, ELO_INITIAL)

        league = m.get("_league_code", "")
        bonus = ELO_HOME_BONUS_BY_LEAGUE.get(league, ELO_HOME_BONUS)
        eh = 1 / (1 + 10 ** ((ra - (rh + bonus)) / 400))
        ea = 1 - eh

        if hg > ag:
            sh, sa = 1.0, 0.0
        elif hg < ag:
            sh, sa = 0.0, 1.0
        else:
            sh, sa = 0.5, 0.5

        goal_diff = abs(hg - ag)
        k_eff = ELO_K * (1 + goal_diff) ** ELO_GOAL_DIFF_EXP

        # Only update the venue-specific rating for each team
        home_ratings[h_id] = rh + k_eff * (sh - eh)
        away_ratings[a_id] = ra + k_eff * (sa - ea)

    return home_ratings, away_ratings
