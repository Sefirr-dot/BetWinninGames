"""
API client for football-data.org (free tier).
Includes automatic rate-limiting and SQLite caching.
"""

import time
import requests
from typing import Optional

import cache
from config import API_KEY, BASE_URL, LEAGUES, HISTORY_SEASONS, RATE_LIMIT_SLEEP


_last_request_time: float = 0.0


def _get(endpoint: str, params: Optional[dict] = None) -> dict:
    """Raw GET with rate-limit throttle."""
    global _last_request_time
    elapsed = time.time() - _last_request_time
    if elapsed < RATE_LIMIT_SLEEP:
        time.sleep(RATE_LIMIT_SLEEP - elapsed)

    url = f"{BASE_URL}{endpoint}"
    headers = {"X-Auth-Token": API_KEY}
    response = requests.get(url, headers=headers, params=params, timeout=30)
    _last_request_time = time.time()

    if response.status_code == 429:
        # Hard rate-limit hit – back off a full minute
        print("  [fetcher] Rate limit hit, sleeping 60s...")
        time.sleep(60)
        return _get(endpoint, params)

    response.raise_for_status()
    return response.json()


def _cached_get(cache_key: str, endpoint: str, params: Optional[dict] = None) -> dict:
    """GET with cache check first."""
    cached = cache.get(cache_key)
    if cached is not None:
        return cached
    data = _get(endpoint, params)
    cache.set(cache_key, data)
    return data


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def get_matches_for_date(date_str: str, league_code: Optional[str] = None) -> list[dict]:
    """
    Return all scheduled/finished matches for a given date (YYYY-MM-DD).
    Optionally filter to a single league.
    """
    leagues = {league_code: LEAGUES[league_code]} if league_code else LEAGUES
    all_matches = []

    for code, comp_id in leagues.items():
        cache_key = f"matches_date_{comp_id}_{date_str}"
        try:
            data = _cached_get(
                cache_key,
                f"/competitions/{comp_id}/matches",
                {"dateFrom": date_str, "dateTo": date_str},
            )
            matches = data.get("matches", [])
            for m in matches:
                m["_league_code"] = code
            all_matches.extend(matches)
        except requests.HTTPError as e:
            print(f"  [fetcher] Warning: could not fetch {code} matches for {date_str}: {e}")

    return all_matches


def get_season_matches(comp_id: int, season: int) -> list[dict]:
    """
    Return all finished matches for a competition season.
    Results are cached indefinitely (no TTL) since past seasons don't change.
    """
    cache_key = f"season_{comp_id}_{season}"
    cached = cache.get_permanent(cache_key)
    if cached is not None:
        return cached

    try:
        data = _get(f"/competitions/{comp_id}/matches", {"season": season})
    except requests.HTTPError as e:
        print(f"  [fetcher] Warning: season {season} for comp {comp_id} unavailable: {e}")
        return []

    finished = [m for m in data.get("matches", []) if m.get("status") == "FINISHED"]
    cache.set(cache_key, finished)
    return finished


def get_standings(comp_id: int) -> list[dict]:
    """Return current standings table entries."""
    cache_key = f"standings_{comp_id}"
    try:
        data = _cached_get(cache_key, f"/competitions/{comp_id}/standings")
        tables = data.get("standings", [])
        for table in tables:
            if table.get("type") == "TOTAL":
                return table.get("table", [])
        return []
    except requests.HTTPError as e:
        print(f"  [fetcher] Warning: standings unavailable for comp {comp_id}: {e}")
        return []


def get_team_matches(team_id: int, limit: int = 30) -> list[dict]:
    """Return recent finished matches for a specific team."""
    cache_key = f"team_matches_{team_id}_{limit}"
    try:
        data = _cached_get(
            cache_key,
            f"/teams/{team_id}/matches",
            {"status": "FINISHED", "limit": limit},
        )
        return data.get("matches", [])
    except requests.HTTPError as e:
        print(f"  [fetcher] Warning: team {team_id} matches unavailable: {e}")
        return []


def get_match(match_id: int) -> dict:
    """
    Fetch a single match by its API ID.
    No cache — always fetches fresh data so results are current.
    """
    return _get(f"/matches/{match_id}")


def load_historical_data(league_code: Optional[str] = None) -> list[dict]:
    """
    Load all historical finished matches across configured seasons.
    Uses cache aggressively.
    """
    leagues = {league_code: LEAGUES[league_code]} if league_code else LEAGUES
    all_matches = []

    for code, comp_id in leagues.items():
        for season in HISTORY_SEASONS:
            print(f"  [fetcher] Loading {code} season {season}...")
            matches = get_season_matches(comp_id, season)
            for m in matches:
                m["_league_code"] = code
            all_matches.extend(matches)

    return all_matches
