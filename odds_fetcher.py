"""
Automatic bookmaker odds fetcher using The Odds API (the-odds-api.com).

Free tier: 500 requests/month.

Fetches best available 1X2 odds across EU bookmakers for a date window
and writes odds/YYYY-MM-DD.csv per day.

Key design decision: one API call per league (not per date), so a full
Fri–Mon weekend window costs only 5 requests total.
"""

import csv
import os
import sqlite3
import time

import requests

from config import ODDS_API_KEY, ODDS_DIR, CACHE_TTL_HOURS

_HISTORY_DB = "cache/odds_history.db"


def _history_conn() -> sqlite3.Connection:
    os.makedirs("cache", exist_ok=True)
    conn = sqlite3.connect(_HISTORY_DB)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS odds_history (
            match_key  TEXT,
            fetched_at REAL,
            odds_1     REAL,
            odds_x     REAL,
            odds_2     REAL,
            PRIMARY KEY (match_key, fetched_at)
        )
    """)
    conn.commit()
    return conn


def save_odds_history(rows_by_date: dict[str, list[dict]]) -> None:
    """Persist fetched odds to odds_history.db for movement tracking."""
    now = time.time()
    with _history_conn() as conn:
        for date_str, rows in rows_by_date.items():
            for row in rows:
                key = f"{row['home_team']}|{row['away_team']}|{date_str}"
                conn.execute(
                    "INSERT OR REPLACE INTO odds_history VALUES (?,?,?,?,?)",
                    (key, now, row["odds_1"], row["odds_x"], row["odds_2"]),
                )
        conn.commit()


def get_odds_movement(home: str, away: str, match_date: str) -> dict | None:
    """
    Return odds movement ratios for a match (earliest / latest odds).

    A ratio > 1.10 on an outcome means the odds shortened significantly
    (sharp money on that outcome).  Returns None when fewer than 2
    historical snapshots exist for this match.

    Returns
    -------
    dict with keys "home", "draw", "away" (float ratios) or None
    """
    if not os.path.exists(_HISTORY_DB):
        return None
    key = f"{home}|{away}|{match_date}"
    with _history_conn() as conn:
        rows = conn.execute(
            "SELECT odds_1, odds_x, odds_2 FROM odds_history "
            "WHERE match_key=? ORDER BY fetched_at ASC",
            (key,),
        ).fetchall()
    if len(rows) < 2:
        return None
    first, last = rows[0], rows[-1]
    # ratio = opening / closing; > 1 means odds shortened (money poured in)
    def _safe_ratio(o, c):
        return round(o / c, 3) if c and c > 1.0 else 1.0
    return {
        "home": _safe_ratio(first[0], last[0]),
        "draw": _safe_ratio(first[1], last[1]),
        "away": _safe_ratio(first[2], last[2]),
    }


def _all_csvs_fresh(dates: list[str]) -> bool:
    """
    Return True if every date CSV already exists and none is older than
    CACHE_TTL_HOURS hours. When True, fetch_window() skips all API calls
    to protect the 500 req/month free-tier quota.
    """
    for d in dates:
        path = os.path.join(ODDS_DIR, f"{d}.csv")
        if not os.path.exists(path):
            return False
        age_h = (time.time() - os.path.getmtime(path)) / 3600
        if age_h > CACHE_TTL_HOURS:
            return False
    return True

_BASE = "https://api.the-odds-api.com/v4"

# The Odds API sport key for each of our league codes
_SPORT_KEYS = {
    "PL":  "soccer_epl",
    "PD":  "soccer_spain_la_liga",
    "BL1": "soccer_germany_bundesliga",
    "FL1": "soccer_france_ligue_one",
}


def _fetch_sport(sport_key: str) -> tuple[list[dict], str]:
    """
    Fetch all upcoming odds for one sport from The Odds API.
    Returns (events_list, quota_remaining_str).
    """
    url = f"{_BASE}/sports/{sport_key}/odds/"
    params = {
        "apiKey":     ODDS_API_KEY,
        "regions":    "eu",
        "markets":    "h2h,totals",   # totals = Over/Under lines (O2.5); same request cost
        "dateFormat": "iso",
        "oddsFormat": "decimal",
    }
    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    remaining = resp.headers.get("x-requests-remaining", "?")
    return resp.json(), remaining


def _best_odds(event: dict) -> dict | None:
    """
    Extract best (highest) 1X2 and Over 2.5 prices across all bookmakers.
    Returns {"odds_1", "odds_x", "odds_2", "odds_o25"} or None if 1X2 incomplete.
    odds_o25 may be None if no bookmaker provides an O2.5 line.
    """
    home = event["home_team"]
    away = event["away_team"]
    best = {"odds_1": 0.0, "odds_x": 0.0, "odds_2": 0.0, "odds_o25": 0.0}

    for bk in event.get("bookmakers", []):
        for mkt in bk.get("markets", []):
            if mkt["key"] == "h2h":
                by_name = {o["name"]: o["price"] for o in mkt["outcomes"]}
                best["odds_1"] = max(best["odds_1"], by_name.get(home, 0.0))
                best["odds_x"] = max(best["odds_x"], by_name.get("Draw", 0.0))
                best["odds_2"] = max(best["odds_2"], by_name.get(away, 0.0))
            elif mkt["key"] == "totals":
                # Find the best Over 2.5 line specifically
                for o in mkt["outcomes"]:
                    if o.get("name") == "Over" and abs(o.get("point", 0) - 2.5) < 0.01:
                        best["odds_o25"] = max(best["odds_o25"], o["price"])

    if min(best["odds_1"], best["odds_x"], best["odds_2"]) <= 1.0:
        return None

    result = {
        "odds_1": best["odds_1"],
        "odds_x": best["odds_x"],
        "odds_2": best["odds_2"],
    }
    if best["odds_o25"] > 1.0:
        result["odds_o25"] = best["odds_o25"]
    return result


def fetch_window(dates: list[str], league_code: str | None = None) -> list[str]:
    """
    Download odds for every date in `dates` (YYYY-MM-DD strings).

    Makes exactly one API call per league regardless of how many dates are
    in the window — the API returns all upcoming events and we filter by date.

    Parameters
    ----------
    dates       : list of date strings, e.g. ["2026-03-06", "2026-03-07", ...]
    league_code : if set, only fetch that league (saves even more quota)

    Returns
    -------
    List of CSV paths written (one per date that had data).
    """
    if not ODDS_API_KEY:
        print("    [odds_fetcher] ODDS_API_KEY no configurado — omitiendo cuotas.")
        return []

    if _all_csvs_fresh(dates):
        print(f"    [odds] CSVs recientes encontrados (<{CACHE_TTL_HOURS}h), omitiendo llamadas API.")
        return []

    leagues = (
        {league_code: _SPORT_KEYS[league_code]}
        if league_code and league_code in _SPORT_KEYS
        else _SPORT_KEYS
    )

    date_set = set(dates)
    by_date: dict[str, list[dict]] = {d: [] for d in dates}

    for code, sport_key in leagues.items():
        try:
            events, remaining = _fetch_sport(sport_key)
            matched = 0
            for ev in events:
                ev_date = ev.get("commence_time", "")[:10]
                if ev_date not in date_set:
                    continue
                odds = _best_odds(ev)
                if odds is None:
                    continue
                row = {
                    "home_team": ev["home_team"],
                    "away_team": ev["away_team"],
                    "odds_1":    round(odds["odds_1"], 2),
                    "odds_x":    round(odds["odds_x"], 2),
                    "odds_2":    round(odds["odds_2"], 2),
                    "odds_o25":  round(odds["odds_o25"], 2) if odds.get("odds_o25") else "",
                }
                by_date[ev_date].append(row)
                matched += 1
            print(f"    [{code}] {matched} partidos con cuotas  "
                  f"(quota restante: {remaining})")
        except requests.HTTPError as e:
            print(f"    [{code}] HTTP error: {e}")
        except Exception as e:
            print(f"    [{code}] Error: {e}")

    os.makedirs(ODDS_DIR, exist_ok=True)
    written = []
    for date_str in sorted(by_date):
        rows = by_date[date_str]
        if not rows:
            continue
        path = os.path.join(ODDS_DIR, f"{date_str}.csv")
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=["home_team", "away_team", "odds_1", "odds_x", "odds_2", "odds_o25"]
            )
            writer.writeheader()
            writer.writerows(rows)
        print(f"    {date_str}: {len(rows)} cuotas guardadas -> {path}")
        written.append(path)

    if not written:
        print("    [odds_fetcher] Sin cuotas para las fechas solicitadas.")
    else:
        save_odds_history(by_date)

    return written
