"""
Referee model.

Referees have statistically stable biases:
  - cards_per_game: some refs give 2× more yellow cards than others
  - home_win_rate: some refs systematically favour home teams
  - penalty_rate: foul-happy refs give more penalties
  - fouls_per_game: affects match flow and late goals

Data source: fdco CSVs have a 'Referee' column.
We build profiles from historical data and use them to:
  1. Improve cards prediction accuracy
  2. Apply a small home/away probability adjustment

Cache: referee stats are stored in cache/referee_stats.json and rebuilt
when fdco data is augmented (lazy re-computation).
"""

from __future__ import annotations

import json
import os

_CACHE_PATH = "cache/referee_stats.json"
_MIN_MATCHES = 10    # minimum matches to trust a referee profile


def build_profiles(matches: list[dict]) -> dict[str, dict]:
    """
    Build referee statistical profiles from a list of match dicts.
    Matches must have '_referee', '_hc' (home cards), '_ac' (away cards),
    '_hp' (home penalty), '_ap' (away penalty) fields when available.

    Returns dict keyed by referee name.
    """
    from collections import defaultdict
    data: dict[str, dict] = defaultdict(lambda: {
        "n": 0, "home_wins": 0, "draws": 0, "away_wins": 0,
        "total_cards": 0.0, "home_cards": 0.0, "away_cards": 0.0,
        "total_fouls": 0.0,
    })

    for m in matches:
        ref = m.get("_referee", "")
        if not ref:
            continue
        d = data[ref]
        d["n"] += 1

        ft = m.get("score", {}).get("fullTime", {})
        hg = ft.get("home")
        ag = ft.get("away")
        if hg is not None and ag is not None:
            if hg > ag:
                d["home_wins"] += 1
            elif hg == ag:
                d["draws"] += 1
            else:
                d["away_wins"] += 1

        d["home_cards"] += m.get("_home_yellow", 0) or 0
        d["away_cards"] += m.get("_away_yellow", 0) or 0
        d["total_cards"] += (m.get("_home_yellow", 0) or 0) + (m.get("_away_yellow", 0) or 0)

    profiles = {}
    for ref, d in data.items():
        n = d["n"]
        if n < _MIN_MATCHES:
            continue
        profiles[ref] = {
            "n":              n,
            "home_win_rate":  round(d["home_wins"] / n, 3),
            "draw_rate":      round(d["draws"] / n, 3),
            "away_win_rate":  round(d["away_wins"] / n, 3),
            "cards_per_game": round(d["total_cards"] / n, 2),
            "home_cards_pg":  round(d["home_cards"] / n, 2),
            "away_cards_pg":  round(d["away_cards"] / n, 2),
        }

    return profiles


def save_profiles(profiles: dict, path: str = _CACHE_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(profiles, f, indent=2)


def load_profiles(path: str = _CACHE_PATH) -> dict:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


# Global league averages (fallback when referee unknown)
_LEAGUE_AVG_CARDS = {
    "PL":  3.8,
    "PD":  5.2,
    "BL1": 3.6,
    "FL1": 4.4,
}
_LEAGUE_AVG_HOME_WIN = {
    "PL":  0.455,
    "PD":  0.480,
    "BL1": 0.460,
    "FL1": 0.465,
}


def get_adjustments(
    referee: str | None,
    league_code: str = "",
    profiles: dict | None = None,
) -> dict:
    """
    Return referee-based adjustments for a match.

    Returns
    -------
    dict with:
        cards_per_game  : expected total cards
        home_bias       : home_win_rate - league_avg (positive = ref favours home)
        home_prob_adj   : small probability adjustment (+/-)
        known           : True if we have data on this referee
    """
    if profiles is None:
        profiles = load_profiles()

    avg_cards    = _LEAGUE_AVG_CARDS.get(league_code, 4.0)
    avg_home_win = _LEAGUE_AVG_HOME_WIN.get(league_code, 0.46)

    if not referee or referee not in profiles:
        return {
            "cards_per_game":  avg_cards,
            "home_bias":       0.0,
            "home_prob_adj":   0.0,
            "known":           False,
        }

    p = profiles[referee]
    home_bias = round(p["home_win_rate"] - avg_home_win, 3)
    # Cap adjustment at ±3% to avoid overcorrecting
    home_prob_adj = max(-0.03, min(0.03, home_bias * 0.5))

    return {
        "cards_per_game":  p["cards_per_game"],
        "home_cards_pg":   p.get("home_cards_pg", avg_cards * 0.5),
        "away_cards_pg":   p.get("away_cards_pg", avg_cards * 0.5),
        "home_bias":       home_bias,
        "home_prob_adj":   round(home_prob_adj, 4),
        "known":           True,
        "n_matches":       p["n"],
    }
