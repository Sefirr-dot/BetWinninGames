"""
Weather data fetcher using Open-Meteo API (free, no API key required).

Fetches temperature, precipitation and wind speed for a stadium at match time.
Used by algorithms/weather_model.py to adjust lambda/mu (expected goals).

Open-Meteo:
  - Free, no authentication
  - 10,000 calls/day
  - Historical + forecast, hourly resolution
  - Endpoint: https://api.open-meteo.com/v1/forecast
"""

import os
import json
import time
import requests
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# Stadium coordinates — all 4 leagues (~80 stadiums)
# Coordinates are for the stadium itself (or city centre when stadium unknown).
# Format: "Team Name as in football-data.org": (latitude, longitude)
# ---------------------------------------------------------------------------

STADIUM_COORDS: dict[str, tuple[float, float]] = {
    # ── Premier League ─────────────────────────────────────────────────────
    "Arsenal FC":             (51.5549,  -0.1084),   # Emirates
    "Chelsea FC":             (51.4816,  -0.1910),   # Stamford Bridge
    "Liverpool FC":           (53.4308,  -2.9608),   # Anfield
    "Manchester United FC":   (53.4631,  -2.2913),   # Old Trafford
    "Manchester City FC":     (53.4831,  -2.2004),   # Etihad
    "Tottenham Hotspur FC":   (51.6043,  -0.0665),   # Tottenham Hotspur Stadium
    "Newcastle United FC":    (54.9756,  -1.6218),   # St James' Park
    "Aston Villa FC":         (52.5092,  -1.8847),   # Villa Park
    "West Ham United FC":     (51.5386,   0.0167),   # London Stadium
    "Brighton & Hove Albion FC": (50.8619, -0.0831), # Amex
    "Fulham FC":              (51.4749,  -0.2217),   # Craven Cottage
    "Brentford FC":           (51.4882,  -0.2886),   # Gtech Community
    "Crystal Palace FC":      (51.3983,  -0.0855),   # Selhurst Park
    "Everton FC":             (53.4388,  -2.9660),   # Goodison Park
    "Wolverhampton Wanderers FC": (52.5900, -2.1302),
    "Nottingham Forest FC":   (52.9400,  -1.1326),
    "Ipswich Town FC":        (52.0553,   1.1451),
    "AFC Bournemouth":        (50.7352,  -1.8381),
    "Southampton FC":         (50.9058,  -1.3912),
    "Leicester City FC":      (52.6204,  -1.1422),
    "Luton Town FC":          (51.8836,  -0.4318),
    "Burnley FC":             (53.7890,  -2.2301),
    "Sheffield United FC":    (53.3703,  -1.4700),

    # ── La Liga ────────────────────────────────────────────────────────────
    "Real Madrid CF":         (40.4531,  -3.6883),   # Bernabeu
    "FC Barcelona":           (41.3809,   2.1228),   # Camp Nou/Spotify Camp Nou
    "Club Atlético de Madrid": (40.4361, -3.5997),   # Civitas Metropolitano
    "Sevilla FC":             (37.3838,  -5.9706),
    "Valencia CF":            (39.4747,  -0.3584),
    "Real Sociedad de Fútbol": (43.3012, -1.9737),
    "Villarreal CF":          (39.9441,  -0.1039),
    "Athletic Club":          (43.2641,  -2.9499),
    "Real Betis Balompié":    (37.3561,  -5.9820),
    "Girona FC":              (41.9833,   2.8012),
    "RC Celta de Vigo":       (42.2121,  -8.7440),
    "RCD Espanyol de Barcelona": (41.3478, 2.0748),
    "Getafe CF":              (40.3259,  -3.7156),
    "Rayo Vallecano de Madrid": (40.3915, -3.6586),
    "UD Almería":             (36.8405,  -2.4469),
    "Real Valladolid CF":     (41.6526,  -4.7328),
    "CA Osasuna":             (42.7966,  -1.6370),
    "RCD Mallorca":           (39.5895,   2.6615),
    "UD Las Palmas":          (28.1074, -15.4308),
    "Deportivo Alavés":       (42.8488,  -2.6881),
    "Leganés":                (40.3288,  -3.7634),

    # ── Bundesliga ─────────────────────────────────────────────────────────
    "FC Bayern München":      (48.2188,  11.6247),   # Allianz Arena
    "Borussia Dortmund":      (51.4926,   7.4519),   # Signal Iduna
    "RB Leipzig":             (51.3457,  12.3483),
    "Bayer 04 Leverkusen":    (51.0374,   6.9836),
    "TSG 1899 Hoffenheim":    (49.2373,   8.8889),
    "SC Freiburg":            (47.9877,   7.8989),
    "Eintracht Frankfurt":    (50.0686,   8.6455),
    "VfL Wolfsburg":          (52.4327,  10.8025),
    "Union Berlin":           (52.4570,  13.5680),
    "Borussia Mönchengladbach": (51.1743, 6.3853),
    "FC Augsburg":            (48.3236,  10.8865),
    "VfB Stuttgart":          (48.7923,   9.2319),
    "Hertha BSC":             (52.5147,  13.2394),
    "FC Köln":                (50.9339,   6.8746),
    "Werder Bremen":          (53.0663,   8.8381),
    "FSV Mainz 05":           (49.9846,   8.2237),
    "VfL Bochum 1848":        (51.4904,   7.2381),
    "1. FC Heidenheim 1846":  (48.6725,  10.1632),
    "SV Darmstadt 98":        (49.8724,   8.6551),
    "FC St. Pauli 1910":      (53.5676,   9.9675),
    "Holstein Kiel":          (54.3459,  10.1222),

    # ── Ligue 1 ────────────────────────────────────────────────────────────
    "Paris Saint-Germain FC": (48.8414,   2.2530),   # Parc des Princes
    "Olympique de Marseille": (43.2696,   5.3967),
    "Olympique Lyonnais":     (45.7659,   4.9820),
    "AS Monaco FC":           (43.7273,   7.4161),
    "Lille OSC":              (50.6119,   3.1301),
    "RC Lens":                (50.4322,   2.8181),
    "OGC Nice":               (43.7050,   7.1926),
    "Stade Rennais FC 1901":  (48.1075,  -1.7145),
    "FC Nantes":              (47.2558,  -1.5251),
    "Montpellier HSC":        (43.6218,   3.8270),
    "Stade de Reims":         (49.2381,   4.0028),
    "RC Strasbourg Alsace":   (48.5598,   7.7521),
    "Toulouse FC":            (43.5836,   1.4339),
    "AJ Auxerre":             (47.7995,   3.5731),
    "Le Havre AC":            (49.4938,   0.1077),
    "Stade Brestois 29":      (48.3939,  -4.4871),
    "AS Saint-Étienne":       (45.4606,   4.3906),
    "Angers SCO":             (47.4667,  -0.5500),
}

_CACHE_FILE  = "cache/weather_cache.json"
_CACHE_TTL_H = 3   # hours before re-fetching same stadium+date


def _load_cache() -> dict:
    try:
        with open(_CACHE_FILE, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save_cache(data: dict) -> None:
    os.makedirs("cache", exist_ok=True)
    with open(_CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f)


def fetch_weather(
    team_name: str,
    match_datetime_utc: str,
) -> dict | None:
    """
    Fetch hourly weather for a team's stadium at match kickoff time.

    Parameters
    ----------
    team_name         : as it appears in football-data.org (e.g. "Arsenal FC")
    match_datetime_utc: ISO string "2026-03-08T15:00:00Z"

    Returns
    -------
    dict with keys:
        temperature_c   : float — 2m temperature (°C)
        wind_speed_kmh  : float — 10m wind speed (km/h)
        precipitation_mm: float — hourly precipitation (mm)
        is_outdoor      : bool  — always True (outdoor stadiums only)
    or None if stadium unknown / API error.
    """
    coords = STADIUM_COORDS.get(team_name)
    if coords is None:
        # Try partial match
        tl = team_name.lower()
        for k, v in STADIUM_COORDS.items():
            if k.lower() in tl or tl in k.lower():
                coords = v
                break
    if coords is None:
        return None

    lat, lon = coords
    try:
        ko = datetime.fromisoformat(match_datetime_utc.replace("Z", "+00:00"))
        date_str = ko.date().isoformat()
        hour     = ko.hour
    except (ValueError, AttributeError):
        return None

    cache_key = f"{lat:.3f},{lon:.3f},{date_str},{hour}"
    cache = _load_cache()

    if cache_key in cache:
        entry = cache[cache_key]
        age_h = (time.time() - entry.get("_fetched_at", 0)) / 3600
        if age_h < _CACHE_TTL_H:
            return {k: v for k, v in entry.items() if not k.startswith("_")}

    try:
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude":           lat,
            "longitude":          lon,
            "hourly":             "temperature_2m,precipitation,wind_speed_10m",
            "start_date":         date_str,
            "end_date":           date_str,
            "wind_speed_unit":    "kmh",
            "timezone":           "UTC",
            "forecast_days":      1,
        }
        resp = requests.get(url, params=params, timeout=8)
        resp.raise_for_status()
        data = resp.json()

        hourly = data.get("hourly", {})
        times  = hourly.get("time", [])
        temps  = hourly.get("temperature_2m", [])
        precip = hourly.get("precipitation", [])
        winds  = hourly.get("wind_speed_10m", [])

        # Find the index closest to kickoff hour
        target_time = f"{date_str}T{hour:02d}:00"
        idx = 0
        for i, t in enumerate(times):
            if t == target_time:
                idx = i
                break

        result = {
            "temperature_c":    round(temps[idx],  1) if idx < len(temps)  else None,
            "wind_speed_kmh":   round(winds[idx],  1) if idx < len(winds)  else None,
            "precipitation_mm": round(precip[idx], 2) if idx < len(precip) else None,
            "is_outdoor":       True,
        }

        cache[cache_key] = {**result, "_fetched_at": time.time()}
        _save_cache(cache)
        return result

    except Exception:
        return None


def fetch_weather_for_match(match: dict) -> dict | None:
    """
    Convenience wrapper: extracts team name and kickoff time from a match dict
    as returned by football-data.org (via fetcher.get_matches_for_date).
    """
    home_name = match.get("homeTeam", {}).get("name", "")
    utc_date  = match.get("utcDate", "")
    if not home_name or not utc_date:
        return None
    return fetch_weather(home_name, utc_date)
