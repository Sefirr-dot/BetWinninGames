"""
Squad depth estimator using football-data.org /teams/{id}/persons endpoint.

Fetches the full squad roster and availability for a team (injuries,
suspensions) to estimate how much attacking/defensive quality is missing.

Unlike lineup_impact.py (which fires < 3h before kickoff with confirmed XI),
squad_depth.py can be run days in advance using the injury/availability data
that football-data.org provides in the /persons endpoint.

Impact model:
  - Missing attackers reduce lambda (expected home goals)
  - Missing key defenders increase mu (expected goals conceded)
  - Impact is scaled by estimated player importance within the squad

Free tier note:
  The football-data.org free tier includes /teams/{id}/persons with role
  and status fields. One call per team, cached 12 hours.

Usage:
    from algorithms.squad_depth import fetch_squad_depth, estimate_depth_impact
    depth = fetch_squad_depth(team_id, api_key)
    impact = estimate_depth_impact(depth)
    # impact = {"lam_mult": 0.95, "mu_mult": 1.02, "notes": [...]}
"""

from __future__ import annotations
import os
import json
import time
import requests

from config import API_KEY, BASE_URL

_CACHE_DIR = "cache/squad_depth"
_CACHE_TTL_H = 12   # re-fetch after 12 hours (injuries change slowly)


def _cache_path(team_id: int) -> str:
    os.makedirs(_CACHE_DIR, exist_ok=True)
    return os.path.join(_CACHE_DIR, f"{team_id}.json")


def fetch_squad_depth(team_id: int) -> dict | None:
    """
    Fetch squad persons data from football-data.org /teams/{id}/persons.

    Returns a dict with player availability info, or None on error.
    Cached for _CACHE_TTL_H hours.
    """
    path = _cache_path(team_id)

    # Check cache
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as f:
                cached = json.load(f)
            age_h = (time.time() - cached.get("_fetched_at", 0)) / 3600
            if age_h < _CACHE_TTL_H:
                return cached
        except Exception:
            pass

    # Fetch from API
    # NOTE: /teams/{id}/persons requires TIER_TWO (paid) on football-data.org.
    # Free tier returns 403 for this endpoint → silently returns None → squad_depth disabled.
    # To enable: upgrade to a paid plan OR populate cache manually.
    try:
        url  = f"{BASE_URL}/teams/{team_id}/persons"
        resp = requests.get(url, headers={"X-Auth-Token": API_KEY}, timeout=10)
        if resp.status_code == 403:
            return None   # free tier — endpoint not available, silently skipped
        if resp.status_code != 200:
            return None
        data = resp.json()
        data["_fetched_at"] = time.time()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)
        return data
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Position importance weights (contribution to team performance)
# ---------------------------------------------------------------------------

_POSITION_LAM_WEIGHT = {
    "Goalkeeper":  0.00,   # GK unavailability doesn't directly reduce attack
    "Defence":     0.05,   # Defenders influence build-up but mainly defence
    "Midfield":    0.10,   # Midfield is crucial for attack creation
    "Offence":     0.18,   # Forwards are the primary attack output
    "Coach":       0.00,
}

_POSITION_MU_WEIGHT = {
    "Goalkeeper":  0.15,   # GK absence increases goals conceded most
    "Defence":     0.12,
    "Midfield":    0.04,
    "Offence":     0.01,
    "Coach":       0.00,
}

# Status strings from football-data.org that indicate unavailability
_UNAVAILABLE_STATUSES = {
    "INJURED", "SUSPENDED", "ON_LOAN", "NATIONAL_DUTY",
    "ABSENT", "UNKNOWN",
}

_AVAILABLE_STATUSES = {
    "AVAILABLE", "IN_TEAM", "ON_BENCH",
}


def estimate_depth_impact(squad_data: dict | None) -> dict:
    """
    Estimate lambda/mu multipliers from squad depth data.

    Parameters
    ----------
    squad_data : from fetch_squad_depth()

    Returns
    -------
    dict:
        lam_mult : float in [0.85, 1.0] — multiplier for expected home goals
        mu_mult  : float in [1.0, 1.15] — multiplier for goals conceded
        depth_ratio : float — fraction of typical squad available (0-1)
        notes    : list[str]
    """
    if not squad_data:
        return _neutral()

    persons = squad_data.get("persons") or squad_data.get("squad", [])
    if not persons:
        return _neutral()

    # Count available vs unavailable by position
    available_by_pos:   dict[str, int] = {}
    unavailable_by_pos: dict[str, int] = {}
    total_players = 0

    for person in persons:
        role     = person.get("role", "")
        position = person.get("position", role)
        status   = (person.get("status") or "AVAILABLE").upper()

        # Normalize position names
        if any(x in position.upper() for x in ("FORWARD", "OFFENCE", "ATTACKER", "STRIKER", "WINGER")):
            pos_key = "Offence"
        elif any(x in position.upper() for x in ("MIDFIELD", "MIDFIELDER")):
            pos_key = "Midfield"
        elif any(x in position.upper() for x in ("DEFENCE", "DEFENDER", "BACK")):
            pos_key = "Defence"
        elif any(x in position.upper() for x in ("GOALKEEPER", "KEEPER", "GK")):
            pos_key = "Goalkeeper"
        elif role == "COACH" or "COACH" in position.upper():
            continue  # skip coaching staff for impact calculation
        else:
            continue  # unknown position — skip

        total_players += 1
        # Explicit unavailable statuses take priority; empty/unknown status = assume available
        # (operator precedence fix: "in A or not in B and C" → "in A or (not in B and C)")
        if status in _UNAVAILABLE_STATUSES or (status not in _AVAILABLE_STATUSES and status != ""):
            unavailable_by_pos[pos_key] = unavailable_by_pos.get(pos_key, 0) + 1
        else:
            # Available, on bench, or unknown/empty status → count as available
            available_by_pos[pos_key] = available_by_pos.get(pos_key, 0) + 1

    if total_players == 0:
        return _neutral()

    # Compute weighted impact
    lam_reduction = 0.0
    mu_increase   = 0.0
    notes         = []

    for pos, n_missing in unavailable_by_pos.items():
        n_avail = available_by_pos.get(pos, 0)
        n_total = n_missing + n_avail
        if n_total == 0:
            continue
        missing_frac = n_missing / n_total

        lam_hit = _POSITION_LAM_WEIGHT[pos] * missing_frac
        mu_hit  = _POSITION_MU_WEIGHT[pos]  * missing_frac

        if lam_hit > 0.005 or mu_hit > 0.005:
            lam_reduction += lam_hit
            mu_increase   += mu_hit
            pos_label = {"Offence": "atacantes", "Midfield": "centrocampistas",
                         "Defence": "defensas", "Goalkeeper": "porteros"}.get(pos, pos)
            notes.append(
                f"{n_missing}/{n_total} {pos_label} no disponibles "
                f"(lam -{lam_hit*100:.1f}%, mu +{mu_hit*100:.1f}%)"
            )

    # Cap adjustments
    lam_reduction = min(lam_reduction, 0.15)
    mu_increase   = min(mu_increase,   0.15)

    # Only report if meaningful impact
    if lam_reduction < 0.01 and mu_increase < 0.01:
        return _neutral()

    # Depth ratio: fraction of typical 25-man squad that is available
    n_unavailable = sum(unavailable_by_pos.values())
    depth_ratio = max(0.0, 1.0 - n_unavailable / max(total_players, 1))

    return {
        "lam_mult":    round(1.0 - lam_reduction, 4),
        "mu_mult":     round(1.0 + mu_increase,   4),
        "depth_ratio": round(depth_ratio,          3),
        "n_unavailable": n_unavailable,
        "n_total_players": total_players,
        "notes":       notes,
    }


def _neutral() -> dict:
    return {
        "lam_mult":        1.0,
        "mu_mult":         1.0,
        "depth_ratio":     1.0,
        "n_unavailable":   0,
        "n_total_players": 0,
        "notes":           [],
    }
