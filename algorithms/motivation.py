"""
Match motivation scorer.

Quantifies how much each team "needs" the result, based on league standings.
Teams fighting for a title, Europa League, or survival play measurably
different football than teams in mid-table dead rubbers.

Returns a motivation_diff (home minus away, range ~-1 to +1):
  > +0.3  → home team much more motivated → λ multiplier boost
  < -0.3  → away team much more motivated → μ multiplier boost
"""

from __future__ import annotations


# Points above/below these thresholds trigger motivation flags
_TITLE_POINTS_GAP     = 6    # within 6 pts of leader → title race
_UCL_POINTS_GAP       = 4    # within 4 pts of top-4 spot → UCL race
_UEL_POINTS_GAP       = 3    # within 3 pts of UEL spot → UEL race
_RELEGATION_POINTS_GAP = 6   # within 6 pts of drop zone → survival battle
_DEAD_RUBBER_GAP      = 15   # 15+ pts clear of both races → low motivation

# Motivation scores per situation (additive)
_SCORE = {
    "title_race":      0.40,
    "ucl_race":        0.30,
    "uel_race":        0.15,
    "survival":        0.45,   # highest — relegation drives max effort
    "dead_rubber":    -0.30,   # low motivation penalty
    "home_boost":      0.05,   # small bonus for home team in any important match
}

# Maximum λ/μ multiplier adjustment from motivation
_MAX_MULT_BOOST = 0.08   # ±8% max effect on expected goals


def score(pos: int, n_teams: int, pts: int,
          leader_pts: int, safe_pts: int, games_left: int) -> float:
    """
    Compute motivation score for one team (0 = neutral, +1 = max urgency).

    Parameters
    ----------
    pos         : current league position (1 = top)
    n_teams     : total teams in league (20 for PL/PD/FL1, 18 for BL1)
    pts         : current points
    leader_pts  : points of league leader
    safe_pts    : points of the first safe-from-relegation team (n_teams - 3)
    games_left  : matches remaining in the season
    """
    if games_left <= 0:
        return 0.0

    s = 0.0
    pts_per_game = 3 * games_left  # max points still available

    # Title race
    gap_to_leader = leader_pts - pts
    if pos <= 4 and gap_to_leader <= _TITLE_POINTS_GAP:
        s += _SCORE["title_race"] * max(0, 1 - gap_to_leader / _TITLE_POINTS_GAP)

    # UCL (top 4 typically)
    ucl_cutoff = 4
    if pos <= ucl_cutoff + 3:
        gap_to_ucl = max(0, (pts - 1) - pts)   # simplified
        if pos <= ucl_cutoff:
            s += _SCORE["ucl_race"] * 0.5       # already in zone, fight to stay
        elif pos <= ucl_cutoff + 2:
            s += _SCORE["ucl_race"]             # chasing top 4

    # Survival (bottom 3 relegated)
    drop_zone = n_teams - 2   # last safe position
    gap_to_safety = safe_pts - pts
    if gap_to_safety <= _RELEGATION_POINTS_GAP and pos >= n_teams - 5:
        urgency = min(1.0, gap_to_safety / max(1, pts_per_game))
        s += _SCORE["survival"] * (1 - urgency + 0.3)  # always high

    # Dead rubber — mathematically safe AND far from title/Europe
    if (gap_to_leader > _DEAD_RUBBER_GAP and
            pos < n_teams - 5 and
            games_left < 10):
        s += _SCORE["dead_rubber"]

    return max(-1.0, min(1.0, s))


def classify_tags(home_score: float, away_score: float) -> list[str]:
    """Return motivation-based context tags."""
    tags = []
    if home_score >= 0.35:
        tags.append("home_must_win")
    if away_score >= 0.35:
        tags.append("away_must_win")
    if home_score >= 0.35 and away_score >= 0.35:
        tags.append("six_pointer")
    if home_score <= -0.20:
        tags.append("home_dead_rubber")
    if away_score <= -0.20:
        tags.append("away_dead_rubber")
    return tags


def goal_multipliers(
    home_score: float,
    away_score: float,
) -> tuple[float, float]:
    """
    Convert motivation scores to λ/μ multipliers.

    A highly motivated team scores ~8% more goals.
    A dead-rubber team scores ~8% fewer.

    Returns (home_mult, away_mult) to be applied to DC lambda/mu.
    """
    diff = home_score - away_score
    home_mult = 1.0 + _MAX_MULT_BOOST * max(-1.0, min(1.0,  diff))
    away_mult = 1.0 + _MAX_MULT_BOOST * max(-1.0, min(1.0, -diff))
    return round(home_mult, 4), round(away_mult, 4)


def from_standings(
    home_pos: int | None,
    away_pos: int | None,
    league_code: str = "",
) -> dict:
    """
    Quick motivation estimate from position only (when full standings unavailable).
    Uses positional heuristics as a fallback.

    Returns dict with home_score, away_score, home_mult, away_mult, tags.
    """
    n = 18 if league_code == "BL1" else 20

    def _pos_score(pos: int) -> float:
        if pos is None:
            return 0.0
        if pos <= 2:
            return 0.35    # title contender
        if pos <= 4:
            return 0.20    # UCL race
        if pos <= 6:
            return 0.10    # UEL race
        if pos >= n - 2:
            return 0.45    # relegation zone
        if pos >= n - 5:
            return 0.25    # near danger
        if 8 <= pos <= n - 6:
            return -0.10   # mid-table drift
        return 0.0

    h = _pos_score(home_pos)
    a = _pos_score(away_pos)
    hm, am = goal_multipliers(h, a)
    tags = classify_tags(h, a)

    return {
        "home_score":  round(h, 3),
        "away_score":  round(a, 3),
        "home_mult":   hm,
        "away_mult":   am,
        "tags":        tags,
    }
