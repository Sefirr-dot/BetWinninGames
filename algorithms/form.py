"""
Recent form analysis with exponential decay.

Computes weighted metrics for the last FORM_WINDOW matches, split by
home/away venue. Returns a combined form score and descriptive string.
"""

from datetime import datetime, date
from config import FORM_DECAY, FORM_WINDOW, ELO_INITIAL


def _parse_date(s: str) -> date:
    try:
        return datetime.strptime(s[:10], "%Y-%m-%d").date()
    except Exception:
        return date.min


def _team_matches(matches: list[dict], team_id: int) -> list[dict]:
    result = []
    for m in matches:
        try:
            if m["homeTeam"]["id"] == team_id or m["awayTeam"]["id"] == team_id:
                hg = m["score"]["fullTime"]["home"]
                ag = m["score"]["fullTime"]["away"]
                if hg is not None and ag is not None:
                    result.append(m)
        except (KeyError, TypeError):
            continue
    return result


def _form_string(win_rate: float) -> str:
    if win_rate >= 0.70:
        return "+++"
    elif win_rate >= 0.55:
        return "++"
    elif win_rate >= 0.40:
        return "+"
    elif win_rate >= 0.25:
        return "-"
    else:
        return "--"


def compute(team_id: int, all_matches: list[dict], venue: str = "all", elo_ratings: dict | None = None) -> dict:
    """
    Compute form metrics for a team.

    venue: "home", "away", or "all"

    Returns:
        points_per_game   weighted points/game (3=W, 1=D, 0=L)
        goals_scored_pg   weighted goals scored per game
        goals_conceded_pg weighted goals conceded per game
        win_rate          weighted win probability
        form_string       visual indicator (e.g. "+++", "--")
        n_matches         number of matches used
    """
    team_ms = _team_matches(all_matches, team_id)

    # Filter by venue
    if venue == "home":
        team_ms = [m for m in team_ms if m["homeTeam"]["id"] == team_id]
    elif venue == "away":
        team_ms = [m for m in team_ms if m["awayTeam"]["id"] == team_id]

    # Sort descending by date (most recent first)
    team_ms.sort(key=lambda m: m.get("utcDate", ""), reverse=True)
    team_ms = team_ms[:FORM_WINDOW]

    if not team_ms:
        return {
            "points_per_game": 1.0,
            "goals_scored_pg": 1.2,
            "goals_conceded_pg": 1.2,
            "win_rate": 0.33,
            "form_string": "-",
            "n_matches": 0,
        }

    total_weight = 0.0
    w_points = 0.0
    w_scored = 0.0
    w_conceded = 0.0
    w_wins = 0.0
    w_xg_scored = 0.0
    w_xg_conceded = 0.0

    for idx, m in enumerate(team_ms):
        w = FORM_DECAY ** idx

        # Strength-of-Schedule: weight this match more if opponent was strong.
        # Uses Elo as a proxy for opponent quality.
        # Quality factor ranges from ~0.6 (very weak opp) to ~1.4 (very strong).
        if elo_ratings:
            is_home_match = m["homeTeam"]["id"] == team_id
            opp_id = m["awayTeam"]["id"] if is_home_match else m["homeTeam"]["id"]
            opp_elo = elo_ratings.get(opp_id, ELO_INITIAL)
            quality_factor = 1.0 + max(-0.4, min(0.4, (opp_elo - ELO_INITIAL) / 1000))
            w *= quality_factor

        is_home = m["homeTeam"]["id"] == team_id
        hg = int(m["score"]["fullTime"]["home"])
        ag = int(m["score"]["fullTime"]["away"])

        scored    = hg if is_home else ag
        conceded  = ag if is_home else hg

        # xG signal: blend 60% actual goals + 40% xG (falls back to actual when absent)
        if is_home:
            xg_for     = m.get("_xg_home", float(scored))
            xg_against = m.get("_xg_away", float(conceded))
        else:
            xg_for     = m.get("_xg_away", float(scored))
            xg_against = m.get("_xg_home", float(conceded))
        eff_for     = 0.6 * scored   + 0.4 * xg_for
        eff_against = 0.6 * conceded + 0.4 * xg_against

        if scored > conceded:
            pts, win = 3, 1
        elif scored == conceded:
            pts, win = 1, 0
        else:
            pts, win = 0, 0

        w_points      += w * pts
        w_scored      += w * scored
        w_conceded    += w * conceded
        w_xg_scored   += w * eff_for
        w_xg_conceded += w * eff_against
        w_wins        += w * win
        total_weight  += w

    if total_weight == 0:
        total_weight = 1e-9

    ppg  = w_points      / total_weight
    spg  = w_scored      / total_weight
    cpg  = w_conceded    / total_weight
    wr   = w_wins        / total_weight

    # Trend: compare recent half vs older half (ascending = "hot", descending = "cold")
    half = max(3, len(team_ms) // 2)
    r_pts = r_w = o_pts = o_w = 0.0
    for idx, m in enumerate(team_ms):
        w = FORM_DECAY ** idx
        is_home = m["homeTeam"]["id"] == team_id
        hg = int(m["score"]["fullTime"]["home"])
        ag = int(m["score"]["fullTime"]["away"])
        scored   = hg if is_home else ag
        conceded = ag if is_home else hg
        pts = 3 if scored > conceded else (1 if scored == conceded else 0)
        if idx < half:
            r_pts += w * pts
            r_w   += w
        else:
            o_pts += w * pts
            o_w   += w
    r_ppg = r_pts / max(r_w, 1e-9)
    o_ppg = o_pts / max(o_w, 1e-9)
    trend = (r_ppg - o_ppg) / 3.0  # normalize to -1..+1

    return {
        "points_per_game":    ppg,
        "goals_scored_pg":    spg,
        "goals_conceded_pg":  cpg,
        "xg_scored_pg":       w_xg_scored   / total_weight,
        "xg_conceded_pg":     w_xg_conceded / total_weight,
        "win_rate":           wr,
        "form_string":        _form_string(wr),
        "n_matches":          len(team_ms),
        "trend":              trend,
    }


def predict(home_id: int, away_id: int, all_matches: list[dict], elo_ratings: dict | None = None) -> dict:
    """
    Derive 1X2 probabilities from form metrics.
    Uses home venue form for home team, away venue form for away team.
    When elo_ratings is provided, each match is weighted by opponent strength (SoS).
    """
    h_form = compute(home_id, all_matches, venue="home", elo_ratings=elo_ratings)
    a_form = compute(away_id, all_matches, venue="away", elo_ratings=elo_ratings)

    def _xg_strength(f: dict) -> float:
        """Normalized xG-based strength: 0..1 range."""
        xg_for = f.get("xg_scored_pg", f["goals_scored_pg"])
        xg_ag  = f.get("xg_conceded_pg", f["goals_conceded_pg"])
        denom  = xg_for + xg_ag + 0.5
        return max(0.01, (xg_for - xg_ag + denom) / (2 * denom))

    h_pts = h_form["points_per_game"] / 3.0  # 0..1
    a_pts = a_form["points_per_game"] / 3.0
    h_trend = h_form.get("trend", 0.0)
    a_trend = a_form.get("trend", 0.0)
    # Blend: 62% points, 33% xG-strength, 5% momentum trend
    h_str = max(0.01, 0.62 * h_pts + 0.33 * _xg_strength(h_form) + 0.05 * h_trend)
    a_str = max(0.01, 0.62 * a_pts + 0.33 * _xg_strength(a_form) + 0.05 * a_trend)

    total = h_str + a_str
    if total < 1e-9:
        h_str = a_str = 0.5
        total = 1.0

    # Raw win shares
    ph = h_str / total
    pa = a_str / total

    # Reserve ~22-28% for draw (decreasing as gap widens)
    gap = abs(ph - pa)
    draw_prob = max(0.10, 0.28 - 0.35 * gap)

    prob_home = ph * (1 - draw_prob)
    prob_away = pa * (1 - draw_prob)
    prob_draw = draw_prob

    s = prob_home + prob_draw + prob_away
    return {
        "prob_home": prob_home / s,
        "prob_draw": prob_draw / s,
        "prob_away": prob_away / s,
        "home_form": h_form,
        "away_form": a_form,
    }
