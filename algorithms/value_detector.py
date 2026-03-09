"""
Value Bet Detector.

Compares ensemble model probabilities against bookmaker implied probabilities
to identify bets with positive edge.

Usage:
    odds_map   = value_detector.load_odds_csv("2026-03-07")
    value_bets = value_detector.find_edges(predictions, odds_map)

CSV format (odds/YYYY-MM-DD.csv):
    home_team,away_team,odds_1,odds_x,odds_2
    Arsenal,Chelsea,2.10,3.40,3.50
"""

import csv
import os
import re
import unicodedata

from config import (ODDS_DIR, VALUE_BET_EDGE_THRESHOLD, VALUE_BET_EDGE_STEP,
                    VALUE_BET_MIN_STARS, VALUE_BET_EDGE_THRESHOLD_BY_LEAGUE,
                    VALUE_BET_MIN_ODDS)

# Adaptive fractional Kelly multipliers by (stars, edge) tier
# Full Kelly only for highest-conviction bets; quarter Kelly as default
def _kelly_fraction(edge: float, bk_odds: float, stars: int) -> float:
    """
    Tiered fractional Kelly staking.

    Tier  | Stars | Edge     | Kelly mult
    ------|-------|----------|------------
    Full  | >= 5  | >= 15%   | 1.0
    Half  | >= 4  | >= 10%   | 0.5
    Qtr   | any   | any      | 0.25

    Hard cap at 25% regardless of tier.
    """
    raw = edge / (bk_odds - 1)
    if stars >= 5 and edge >= 0.15:
        mult = 1.0
    elif stars >= 4 and edge >= 0.10:
        mult = 0.5
    else:
        mult = 0.25
    return min(raw * mult, 0.25)

# Common prefixes/suffixes and article words stripped during normalization
_PREFIXES = ("1. fc ", "fc ", "cf ", "rc ", "ac ", "sc ", "as ", "afc ",
             "club ", "ca ", "cd ", "sd ", "ud ", "vfl ", "vfb ", "fsv ",
             "rcd ", "rsc ", "bsc ", "fk ", "sk ", "nk ")
_SUFFIXES = (" fc", " cf", " afc", " sc", " ac", " utd", " united",
             " city", " town", " athletic", " athletico", " sv", " rv")
_ARTICLES = (" de ", " del ", " la ", " el ", " los ", " las ",
             " di ", " du ", " le ", " les ", " da ", " do ")
# City name aliases: NFKD transliteration → English common name
_ALIASES = {"munchen": "munich", "koeln": "cologne", "koln": "cologne"}


def _normalize(name: str) -> str:
    """
    Robust team name normalisation.
    1. Replace hyphens with spaces.
    2. Transliterate accented chars via NFKD (ü→u, é→e, ö→o, …).
    3. Strip common prefixes (FC, RC, Club, …) and suffixes (FC, Utd, …).
    4. Remove common article words (de, del, la, …) from the middle.
    5. Strip non-alphanumeric chars and collapse whitespace.
    6. Apply city aliases (munchen→munich).
    """
    n = name.lower().strip()
    n = n.replace("-", " ").replace("_", " ")
    # Transliterate accented characters
    n = unicodedata.normalize("NFKD", n).encode("ascii", "ignore").decode("ascii")
    # Strip common prefixes (first match only)
    for prefix in _PREFIXES:
        if n.startswith(prefix):
            n = n[len(prefix):]
            break
    # Strip common suffixes (first match only)
    for suffix in _SUFFIXES:
        if n.endswith(suffix):
            n = n[: -len(suffix)].strip()
            break
    # Remove article words
    for word in _ARTICLES:
        n = n.replace(word, " ")
    # Strip non-alphanumeric except spaces, collapse whitespace
    n = re.sub(r"[^a-z0-9 ]", "", n)
    n = re.sub(r"\s+", " ", n).strip()
    # Apply city aliases
    for old, new in _ALIASES.items():
        n = re.sub(r"\b" + old + r"\b", new, n)
    return n


def _sig_tokens(s: str) -> set:
    """Significant tokens: words with ≥ 4 chars (excludes short abbreviations)."""
    return {t for t in s.split() if len(t) >= 4}


def load_odds_csv(date_str: str) -> dict:
    """
    Load bookmaker odds from odds/YYYY-MM-DD.csv.

    Returns a dict keyed by (norm_home, norm_away) ->
        {odds_1, odds_x, odds_2, home_team, away_team}.
    Returns {} if the file doesn't exist.
    """
    path = os.path.join(ODDS_DIR, f"{date_str}.csv")
    if not os.path.exists(path):
        return {}

    odds_map = {}
    try:
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                home_raw = row.get("home_team", "").strip()
                away_raw = row.get("away_team", "").strip()
                if not home_raw or not away_raw:
                    continue
                try:
                    entry = {
                        "odds_1":     float(row["odds_1"]),
                        "odds_x":     float(row["odds_x"]),
                        "odds_2":     float(row["odds_2"]),
                        "home_team":  home_raw,
                        "away_team":  away_raw,
                    }
                    # Optional O2.5 / BTTS columns — present when odds_fetcher fetches totals market
                    if row.get("odds_o25"):
                        try:
                            entry["odds_o25"] = float(row["odds_o25"])
                        except ValueError:
                            pass
                    if row.get("odds_btts"):
                        try:
                            entry["odds_btts"] = float(row["odds_btts"])
                        except ValueError:
                            pass
                    odds_map[(_normalize(home_raw), _normalize(away_raw))] = entry
                except (KeyError, ValueError):
                    continue
    except Exception as e:
        print(f"  [value_detector] Error loading {path}: {e}")

    return odds_map


def _match_odds(home: str, away: str, odds_map: dict) -> dict | None:
    """
    Look up odds for a match using three-tier matching:
    1. Exact normalised key.
    2. Substring containment.
    3. Significant-token overlap (handles year suffixes like "TSG 1899 Hoffenheim").
    """
    norm_home = _normalize(home)
    norm_away = _normalize(away)

    # 1. Exact match
    key = (norm_home, norm_away)
    if key in odds_map:
        return odds_map[key]

    # 2. Substring fallback
    for (h_key, a_key), entry in odds_map.items():
        if (norm_home in h_key or h_key in norm_home) and \
           (norm_away in a_key or a_key in norm_away):
            return entry

    # 3. Token overlap fallback
    t_home = _sig_tokens(norm_home)
    t_away = _sig_tokens(norm_away)
    for (h_key, a_key), entry in odds_map.items():
        t_h = _sig_tokens(h_key)
        t_a = _sig_tokens(a_key)
        h_match = bool(t_home) and bool(t_h) and (t_home <= t_h or t_h <= t_home)
        a_match = bool(t_away) and bool(t_a) and (t_away <= t_a or t_a <= t_away)
        if h_match and a_match:
            return entry

    return None


def get_match_odds(home: str, away: str, odds_map: dict) -> dict | None:
    """
    Look up bookmaker odds for a specific match from a pre-loaded odds_map.

    Returns dict with keys ``odds_1``, ``odds_x``, ``odds_2`` or None if
    the match is not found. Public wrapper for the internal ``_match_odds()``
    so callers (e.g. ensemble.py via main.py) can retrieve per-match odds
    without going through ``find_edges()``.
    """
    return _match_odds(home, away, odds_map)


def find_edges(predictions: list[dict], odds_map: dict) -> list[dict]:
    """
    Identify value bets where model probability exceeds bookmaker implied
    probability by at least VALUE_BET_EDGE_THRESHOLD.

    Parameters
    ----------
    predictions : ranked list from ensemble.rank_predictions()
    odds_map    : from load_odds_csv()

    Returns
    -------
    List of dicts sorted by edge DESC, each containing:
        match, league, outcome, model_prob, implied_prob,
        edge, bk_odds, kelly_fraction, home_name, away_name
    """
    if not odds_map:
        return []

    value_bets = []

    for entry in predictions:
        mi   = entry.get("match_info", {})
        pred = entry.get("prediction", {})

        home_name = mi.get("homeTeam", {}).get("name", "")
        away_name = mi.get("awayTeam", {}).get("name", "")

        odds = _match_odds(home_name, away_name, odds_map)
        if odds is None:
            continue

        stars = pred.get("stars", 3)
        if stars < VALUE_BET_MIN_STARS:
            continue
        league = mi.get("_league_code", "")
        base_threshold = VALUE_BET_EDGE_THRESHOLD_BY_LEAGUE.get(league, VALUE_BET_EDGE_THRESHOLD)
        effective_threshold = base_threshold + (5 - stars) * VALUE_BET_EDGE_STEP

        checks = [
            ("home",  pred.get("prob_home", 0.0), odds["odds_1"]),
            ("draw",  pred.get("prob_draw", 0.0), odds["odds_x"]),
            ("away",  pred.get("prob_away", 0.0), odds["odds_2"]),
        ]

        # Add Over 2.5 and BTTS when bookmaker odds are available in the CSV
        if (odds.get("odds_o25") or 0) > 1.0:
            checks.append(("over25", pred.get("over25", 0.0), odds["odds_o25"]))
        if (odds.get("odds_btts") or 0) > 1.0:
            checks.append(("btts", pred.get("btts_prob", 0.0), odds["odds_btts"]))

        for outcome, model_prob, bk_odds in checks:
            if bk_odds <= VALUE_BET_MIN_ODDS:
                continue  # market too efficient at short odds — skip
            implied_prob = 1.0 / bk_odds
            edge = model_prob - implied_prob

            if edge >= effective_threshold:
                kelly = _kelly_fraction(edge, bk_odds, stars)
                value_bets.append({
                    "match":          f"{home_name} vs {away_name}",
                    "league":         mi.get("_league_code", ""),
                    "outcome":        outcome,
                    "model_prob":     model_prob,
                    "implied_prob":   implied_prob,
                    "edge":           edge,
                    "bk_odds":        bk_odds,
                    "kelly_fraction": kelly,
                    "home_name":      home_name,
                    "away_name":      away_name,
                })

    return sorted(value_bets, key=lambda x: x["edge"], reverse=True)
