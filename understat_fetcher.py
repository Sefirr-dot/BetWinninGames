"""
Understat xG scraper for BetWinninGames.

Fetches expected-goals (xG) data for all 5 supported leagues and enriches
historical match dicts with _xg_home / _xg_away fields consumed by
dixon_coles.fit() during model training.

Caches all data in cache/understat_xg.db — each (league, season) pair is
fetched only once and stored indefinitely.
"""

import json
import os
import sqlite3
import time
from collections import defaultdict
from datetime import datetime
from difflib import SequenceMatcher

import requests

from config import UNDERSTAT_LEAGUES, UNDERSTAT_XG_DB, UNDERSTAT_SEASONS

_UNDERSTAT_BASE = "https://understat.com/league"
_UNDERSTAT_API  = "https://understat.com/getLeagueData"  # GET /{league}/{season}
_REQUEST_DELAY  = 2.5   # seconds between requests (be polite to Understat)


# ---------------------------------------------------------------------------
# SQLite cache
# ---------------------------------------------------------------------------

def _connect() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(UNDERSTAT_XG_DB), exist_ok=True)
    return sqlite3.connect(UNDERSTAT_XG_DB)


def _init_db() -> None:
    with _connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS xg_matches (
                league     TEXT    NOT NULL,
                season     INTEGER NOT NULL,
                match_date TEXT    NOT NULL,
                home_name  TEXT    NOT NULL,
                away_name  TEXT    NOT NULL,
                xg_home    REAL    NOT NULL,
                xg_away    REAL    NOT NULL,
                goals_home INTEGER NOT NULL,
                goals_away INTEGER NOT NULL,
                PRIMARY KEY (league, season, match_date, home_name, away_name)
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS fetched_seasons (
                league TEXT    NOT NULL,
                season INTEGER NOT NULL,
                PRIMARY KEY (league, season)
            )
        """)
        conn.commit()


def _is_cached(league: str, season: int) -> bool:
    with _connect() as conn:
        row = conn.execute(
            "SELECT 1 FROM fetched_seasons WHERE league=? AND season=?",
            (league, season),
        ).fetchone()
    return row is not None


def _save_to_cache(league: str, season: int, matches: list[dict]) -> None:
    with _connect() as conn:
        for m in matches:
            conn.execute(
                """INSERT OR IGNORE INTO xg_matches
                   (league, season, match_date, home_name, away_name,
                    xg_home, xg_away, goals_home, goals_away)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                (
                    league, season,
                    m["match_date"], m["home_name"], m["away_name"],
                    m["xg_home"], m["xg_away"],
                    m["goals_home"], m["goals_away"],
                ),
            )
        conn.execute(
            "INSERT OR IGNORE INTO fetched_seasons VALUES (?,?)",
            (league, season),
        )
        conn.commit()


def _load_from_cache(league: str, season: int) -> list[dict]:
    with _connect() as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """SELECT match_date, home_name, away_name,
                      xg_home, xg_away, goals_home, goals_away
               FROM xg_matches WHERE league=? AND season=?""",
            (league, season),
        ).fetchall()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Understat AJAX API
# ---------------------------------------------------------------------------

def _fetch_via_api(league: str, season: int) -> list[dict]:
    """
    Fetch match data from Understat's JSON API.

    The page shell loads via league.min.js which GETs:
      https://understat.com/getLeagueData/{league}/{season}

    Response is a JSON object with keys: teams, players, dates.
    The 'dates' list contains one entry per match with isResult, h/a, goals, xG.
    """
    understat_league = UNDERSTAT_LEAGUES[league]
    url = f"{_UNDERSTAT_API}/{understat_league}/{season}"
    resp = requests.get(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "X-Requested-With": "XMLHttpRequest",
            "Referer": f"{_UNDERSTAT_BASE}/{understat_league}/{season}",
            "Accept": "application/json, text/javascript, */*; q=0.01",
        },
        timeout=30,
    )
    resp.raise_for_status()

    data = resp.json()
    return _parse_match_entries(data.get("dates", []))


def _parse_match_entries(entries: list) -> list[dict]:
    """Convert raw Understat entry dicts to our internal format (matches DB column names)."""
    matches = []
    for entry in entries:
        if not entry.get("isResult"):
            continue
        try:
            matches.append({
                "match_date": entry["datetime"][:10],
                "home_name":  entry["h"]["title"],
                "away_name":  entry["a"]["title"],
                "xg_home":    float(entry["xG"]["h"]),
                "xg_away":    float(entry["xG"]["a"]),
                "goals_home": int(entry["goals"]["h"]),
                "goals_away": int(entry["goals"]["a"]),
            })
        except (KeyError, TypeError, ValueError):
            continue
    return matches


def fetch_league_xg(league: str, season: int) -> list[dict]:
    """
    Return xG match data for one league+season, using the SQLite cache.
    Fetches from Understat on first call; subsequent calls are instant.
    Returns [] on any scraping error.
    """
    _init_db()

    if _is_cached(league, season):
        return _load_from_cache(league, season)

    print(f"    [xG] Descargando {league} {season} desde Understat...")
    try:
        matches = _fetch_via_api(league, season)
        if matches:
            _save_to_cache(league, season, matches)
            print(f"    [xG] {league} {season}: {len(matches)} partidos cacheados.")
        else:
            print(f"    [xG] {league} {season}: sin datos (página vacía o formato inesperado).")
        time.sleep(_REQUEST_DELAY)
        return matches
    except Exception as exc:
        print(f"    [xG] WARNING: {league} {season} no disponible: {exc}")
        return []


# ---------------------------------------------------------------------------
# Team name normalisation & fuzzy matching
# ---------------------------------------------------------------------------

def _normalize(name: str) -> str:
    """Lowercase + strip common legal suffixes for fuzzy comparison."""
    n = name.lower().strip()
    for suffix in (" fc", " cf", " ac", " sc", " afc", " fk", " sk", " united", " city"):
        if n.endswith(suffix):
            n = n[: -len(suffix)].strip()
    return n


def _fuzzy_match(target: str, candidates: list[str], threshold: float = 0.65) -> str | None:
    """Return the candidate with the highest SequenceMatcher ratio ≥ threshold."""
    norm_t = _normalize(target)
    best_r, best_c = 0.0, None
    for c in candidates:
        r = SequenceMatcher(None, norm_t, _normalize(c)).ratio()
        if r > best_r:
            best_r, best_c = r, c
    return best_c if best_r >= threshold else None


# ---------------------------------------------------------------------------
# Season helper
# ---------------------------------------------------------------------------

def _season_year(date_str: str) -> int:
    """Return the season start year (Aug 2024 → 2024, Jan 2025 → 2024)."""
    d = datetime.strptime(date_str[:10], "%Y-%m-%d")
    return d.year if d.month >= 7 else d.year - 1


# ---------------------------------------------------------------------------
# Main enrichment function (called from main.py and backtest.py)
# ---------------------------------------------------------------------------

def enrich_with_xg(fd_matches: list[dict]) -> None:
    """
    Enrich fd_matches IN-PLACE with _xg_home / _xg_away fields.

    Strategy:
    1. Group fd matches by (league_code, season).
    2. For each group, fetch the corresponding Understat xG data.
    3. Build an index: (date, goals_home, goals_away) → list[xg_match].
    4. For each fd match:
       - If exactly 1 Understat match for that key → direct match, done.
       - If >1 (same day + same score) → use fuzzy team name to disambiguate.
       - If 0 → leave unchanged (Dixon-Coles falls back to actual goals).
    """
    # --- Group fd matches by (league, season) ---
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for m in fd_matches:
        league = m.get("_league_code")
        if not league or league not in UNDERSTAT_LEAGUES:
            continue
        utc = m.get("utcDate", "")
        if not utc:
            continue
        try:
            gh = m["score"]["fullTime"]["home"]
            ga = m["score"]["fullTime"]["away"]
            if gh is None or ga is None:
                continue
        except (KeyError, TypeError):
            continue
        season = _season_year(utc)
        if season not in UNDERSTAT_SEASONS:
            continue
        groups[(league, season)].append(m)

    total_enriched = 0

    for (league, season), group in groups.items():
        xg_list = fetch_league_xg(league, season)
        if not xg_list:
            continue

        # Build lookup: (date, goals_h, goals_a) → [xg_match, ...]
        by_score: dict[tuple, list[dict]] = defaultdict(list)
        for xm in xg_list:
            key = (xm["match_date"], xm["goals_home"], xm["goals_away"])
            by_score[key].append(xm)

        for m in group:
            date_str = m["utcDate"][:10]
            gh       = int(m["score"]["fullTime"]["home"])
            ga       = int(m["score"]["fullTime"]["away"])
            key      = (date_str, gh, ga)
            cands    = by_score.get(key, [])

            if len(cands) == 1:
                m["_xg_home"] = cands[0]["xg_home"]
                m["_xg_away"] = cands[0]["xg_away"]
                total_enriched += 1

            elif len(cands) > 1:
                # Ambiguous scoreline — use fuzzy team name
                home_name  = m.get("homeTeam", {}).get("name", "")
                cand_homes = [c["home_name"] for c in cands]
                matched    = _fuzzy_match(home_name, cand_homes)
                if matched:
                    xm = next(c for c in cands if c["home_name"] == matched)
                    m["_xg_home"] = xm["xg_home"]
                    m["_xg_away"] = xm["xg_away"]
                    total_enriched += 1

    total_eligible = sum(
        1 for m in fd_matches
        if m.get("_league_code") in UNDERSTAT_LEAGUES
        and m.get("score", {}).get("fullTime", {}).get("home") is not None
    )
    print(f"    [xG] {total_enriched}/{total_eligible} partidos enriquecidos con xG.")
