"""
Head-to-head historical record between two specific teams.

Uses time-based exponential decay (H2H_YEARLY_DECAY per year of elapsed time)
so that a match from 2 years ago is weighted at H2H_YEARLY_DECAY^2 regardless
of how many other H2H matches exist. This is more robust than index-based decay
when fixtures are clustered or sparse over time.

Only activated when there are enough H2H matches (H2H_MIN_MATCHES).
"""

from datetime import date as _date, datetime as _datetime

from config import FORM_DECAY, H2H_MIN_MATCHES, H2H_YEARLY_DECAY


def predict(
    home_id: int,
    away_id: int,
    all_matches: list[dict],
    reference_date: _date | None = None,
) -> dict:
    """
    Compute win/draw/loss probabilities from direct H2H history.

    All matches between the two teams are considered regardless of venue,
    but results are expressed from home_id's current-match perspective
    (i.e. home_id is playing at home today).

    When reference_date is provided (recommended), weights are:
        w = H2H_YEARLY_DECAY ** (days_elapsed / 365)
    so a match 1 year ago = 0.70×, 2 years ago = 0.49×, 3 years = 0.34×.
    Falls back to index-based FORM_DECAY if a match has no parseable date.

    Returns
    -------
    sufficient : bool   — False when n < H2H_MIN_MATCHES (model not used)
    n_matches  : int
    prob_home, prob_draw, prob_away
    btts_rate  : float  — fraction of H2H matches where both scored
    avg_goals  : float  — weighted average total goals per H2H match
    home_wins, draws, away_wins : int  (raw counts for display)
    """
    h2h = []
    for m in all_matches:
        try:
            hg = m["score"]["fullTime"]["home"]
            ag = m["score"]["fullTime"]["away"]
            if hg is None or ag is None:
                continue
            h = m["homeTeam"]["id"]
            a = m["awayTeam"]["id"]
            if (h == home_id and a == away_id) or (h == away_id and a == home_id):
                h2h.append(m)
        except (KeyError, TypeError):
            continue

    n = len(h2h)
    if n < H2H_MIN_MATCHES:
        return {"sufficient": False, "n_matches": n}

    # Most recent first
    h2h.sort(key=lambda m: m.get("utcDate", ""), reverse=True)

    ref = reference_date or _date.today()

    total_w = w_home = w_draw = w_away = w_btts = w_goals = 0.0
    raw_home = raw_draw = raw_away = 0

    for idx, m in enumerate(h2h):
        # Time-based decay: H2H_YEARLY_DECAY ^ (days_elapsed / 365)
        utc = m.get("utcDate", "")
        try:
            match_date = _datetime.strptime(utc[:10], "%Y-%m-%d").date()
            days_elapsed = max(0, (ref - match_date).days)
            w = H2H_YEARLY_DECAY ** (days_elapsed / 365.0)
        except (ValueError, TypeError):
            # No parseable date — fall back to index-based decay
            w = FORM_DECAY ** idx

        h = m["homeTeam"]["id"]
        hg = int(m["score"]["fullTime"]["home"])
        ag = int(m["score"]["fullTime"]["away"])

        # Express result from today's home team perspective
        if h == home_id:
            scored, conceded = hg, ag
        else:
            scored, conceded = ag, hg

        if scored > conceded:
            w_home += w
            raw_home += 1
        elif scored == conceded:
            w_draw += w
            raw_draw += 1
        else:
            w_away += w
            raw_away += 1

        if hg > 0 and ag > 0:
            w_btts += w
        w_goals += w * (hg + ag)
        total_w += w

    if total_w < 1e-9:
        return {"sufficient": False, "n_matches": n}

    ph = w_home / total_w
    pd = w_draw / total_w
    pa = w_away / total_w
    s = ph + pd + pa

    return {
        "sufficient": True,
        "n_matches": n,
        "prob_home": ph / s,
        "prob_draw": pd / s,
        "prob_away": pa / s,
        "btts_rate": w_btts / total_w,
        "avg_goals": w_goals / total_w,
        "home_wins": raw_home,
        "draws": raw_draw,
        "away_wins": raw_away,
    }
