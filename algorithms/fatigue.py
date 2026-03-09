"""
Fatigue factor based on days of rest since the team's last match.

Applied as a multiplicative adjustment to expected goals (lambda/mu) inside
the Dixon-Coles prediction, so a fatigued team is expected to score less.

Calibration (approximate, based on published sports science literature):
  >=7 days  → 1.00 (fully recovered)
    6 days  → 0.99
    5 days  → 0.97
    4 days  → 0.94
    3 days  → 0.90
    2 days  → 0.86
   <=1 day  → 0.82
"""

from datetime import datetime, date

# Days → expected-goals multiplier
_CURVE: dict[int, float] = {
    0: 0.82,
    1: 0.82,
    2: 0.86,
    3: 0.90,
    4: 0.94,
    5: 0.97,
    6: 0.99,
}


def days_rest(team_id: int, all_matches: list[dict], ref: date) -> int | None:
    """
    How many days since `team_id` last played a FINISHED match before `ref`.
    Returns None if no such match is found.
    """
    last = None
    for m in all_matches:
        try:
            if m["homeTeam"]["id"] != team_id and m["awayTeam"]["id"] != team_id:
                continue
            if m.get("status") != "FINISHED":
                continue
            d = datetime.strptime(m["utcDate"][:10], "%Y-%m-%d").date()
            if d < ref and (last is None or d > last):
                last = d
        except (KeyError, TypeError, ValueError):
            continue
    return None if last is None else (ref - last).days


def multiplier(days: int | None) -> float:
    """Expected-goals multiplier for a team with `days` days of rest."""
    if days is None or days >= 7:
        return 1.0
    return _CURVE.get(max(0, days), 1.0)


def compute(
    home_id: int,
    away_id: int,
    all_matches: list[dict],
    ref: date,
) -> dict:
    """
    Return fatigue info for both teams.

    Returns
    -------
    home_days   : int | None
    away_days   : int | None
    home_mult   : float   (1.0 = no penalty)
    away_mult   : float
    """
    h_days = days_rest(home_id, all_matches, ref)
    a_days = days_rest(away_id, all_matches, ref)
    return {
        "home_days": h_days,
        "away_days": a_days,
        "home_mult": multiplier(h_days),
        "away_mult": multiplier(a_days),
    }
