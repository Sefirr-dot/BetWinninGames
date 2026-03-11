"""
Lineup impact estimator.

When starting lineups are announced (typically 60-75 min before kickoff),
this module:
  1. Fetches the confirmed lineup from football-data.org /matches/{id}
  2. Compares it against the "typical" XI (most frequent starters this season)
  3. Estimates the probability impact of missing key players

Impact is modelled as a reduction to lambda/mu based on:
  - Positional heuristics (missing attackers hurt attack, missing defenders
    hurt defence)
  - Team xG profile from understat_xg.db: attack-reliant teams (high xG-for)
    are penalised more for losing attackers; defence-reliant teams
    (high xG-against) are penalised more for losing defenders.

This runs automatically when main.py detects that a match starts
within the next 3 hours and lineups are available.
"""

from __future__ import annotations
import os
import sqlite3

import requests

from config import API_KEY, BASE_URL, UNDERSTAT_XG_DB

_HEADERS = {"X-Auth-Token": API_KEY}
_TIMEOUT = 10

# Estimated λ/μ reduction per missing player by position (base values)
# Scaled up/down based on team xG profile loaded from understat_xg.db
_IMPACT = {
    "Goalkeeper":  {"lam": 0.00, "mu": 0.08},   # missing GK → concede more
    "Defender":    {"lam": 0.02, "mu": 0.06},
    "Midfielder":  {"lam": 0.05, "mu": 0.03},
    "Attacker":    {"lam": 0.10, "mu": 0.01},
}
_MAX_TOTAL_IMPACT = 0.20   # cap total adjustment at ±20%

# League average xG/match (baseline for scaling)
_LEAGUE_XG_BASELINE = {
    "PL":  1.20,
    "PD":  1.10,
    "BL1": 1.25,
    "FL1": 1.05,
}


def fetch_lineup(match_id: int) -> dict | None:
    """
    Fetch lineup data from football-data.org /matches/{id}.
    Returns {"home": [...players...], "away": [...players...]} or None.
    """
    try:
        url = f"{BASE_URL}/matches/{match_id}"
        resp = requests.get(url, headers=_HEADERS, timeout=_TIMEOUT)
        if resp.status_code != 200:
            return None
        data = resp.json()
        lineups = data.get("lineups", [])
        if len(lineups) < 2:
            return None
        result = {}
        for lineup in lineups:
            side = "home" if lineup.get("type") == "HOME" else "away"
            result[side] = [
                {
                    "name":     p.get("name", ""),
                    "position": p.get("position", ""),
                    "shirt":    p.get("shirtNumber"),
                }
                for p in lineup.get("startXI", [])
            ]
        return result if "home" in result and "away" in result else None
    except Exception:
        return None


def _load_team_xg_profile(
    team_name: str,
    league: str,
    xg_db: str = UNDERSTAT_XG_DB,
    n_matches: int = 20,
) -> dict:
    """
    Load team's average xG-for and xG-against from the last n_matches
    stored in understat_xg.db.

    Returns {"xg_for_pg": float, "xg_against_pg": float} or {} on failure.
    Used to scale positional impact: attack-heavy teams lose more when missing
    attackers; defence-heavy teams lose more when missing defenders.
    """
    if not os.path.exists(xg_db):
        return {}
    try:
        with sqlite3.connect(xg_db) as conn:
            # Home matches
            home_rows = conn.execute(
                "SELECT xg_home, xg_away FROM xg_matches "
                "WHERE home_name = ? ORDER BY match_date DESC LIMIT ?",
                (team_name, n_matches),
            ).fetchall()
            # Away matches
            away_rows = conn.execute(
                "SELECT xg_away, xg_home FROM xg_matches "
                "WHERE away_name = ? ORDER BY match_date DESC LIMIT ?",
                (team_name, n_matches),
            ).fetchall()

        all_for     = [r[0] for r in home_rows] + [r[0] for r in away_rows]
        all_against = [r[1] for r in home_rows] + [r[1] for r in away_rows]
        if not all_for:
            return {}

        return {
            "xg_for_pg":     round(sum(all_for)     / len(all_for),     3),
            "xg_against_pg": round(sum(all_against) / len(all_against), 3),
        }
    except Exception:
        return {}


def estimate_impact(
    lineup: dict | None,
    typical_lineup: dict | None = None,
    home_team: str = "",
    away_team: str = "",
    league: str = "",
) -> dict:
    """
    Estimate lambda/mu multipliers from lineup changes.

    Uses a positional heuristic scaled by each team's xG profile:
      - Attack-reliant teams (xG-for >> league avg) lose more λ when short of attackers.
      - Defence-reliant teams (xG-against >> league avg) concede more μ when short of defenders.

    Parameters
    ----------
    lineup         : {"home": [...], "away": [...]} from fetch_lineup()
    typical_lineup : optional reference XI (not currently used — reserved for future)
    home_team      : team name for xG profile lookup (matched against understat DB)
    away_team      : team name for xG profile lookup
    league         : league code (PL/PD/BL1/FL1) for baseline xG reference

    Returns
    -------
    dict with "home_lam_mult", "away_mu_mult", "notes", "home_xi", "away_xi"
    """
    if not lineup:
        return {"home_lam_mult": 1.0, "away_mu_mult": 1.0, "notes": [],
                "home_xi": [], "away_xi": []}

    notes = []
    baseline_xg = _LEAGUE_XG_BASELINE.get(league, 1.15)

    # Load xG profiles for both teams (graceful fallback to empty dict)
    home_profile = _load_team_xg_profile(home_team, league) if home_team else {}
    away_profile = _load_team_xg_profile(away_team, league) if away_team else {}

    # Scaling factors: how much more (or less) to penalise based on team style
    # A team with xG-for = 1.6 vs baseline 1.2 → attack_scale = 1 + (1.6-1.2)*0.25 = 1.10
    home_attack_scale  = 1.0 + max(0, (home_profile.get("xg_for_pg",     baseline_xg) - baseline_xg) * 0.25)
    home_defence_scale = 1.0 + max(0, (home_profile.get("xg_against_pg", baseline_xg) - baseline_xg) * 0.25)
    away_attack_scale  = 1.0 + max(0, (away_profile.get("xg_for_pg",     baseline_xg) - baseline_xg) * 0.25)
    away_defence_scale = 1.0 + max(0, (away_profile.get("xg_against_pg", baseline_xg) - baseline_xg) * 0.25)

    def _count_attackers(players: list[dict]) -> int:
        return sum(1 for p in players if p.get("position") in ("Attacker", "Forward"))

    def _count_defenders(players: list[dict]) -> int:
        return sum(1 for p in players if p.get("position") in ("Defender",))

    home_players = lineup.get("home", [])
    away_players = lineup.get("away", [])

    home_att = _count_attackers(home_players)
    away_att = _count_attackers(away_players)
    home_def = _count_defenders(home_players)
    away_def = _count_defenders(away_players)

    home_lam_mult = 1.0
    away_mu_mult  = 1.0

    # Few attackers → reduced attack λ (scaled by how attack-reliant the team is)
    if home_att < 2:
        penalty = 0.08 * home_attack_scale
        home_lam_mult -= penalty
        notes.append(f"Local con pocos delanteros ({home_att}), penalizacion lam -{penalty:.2f}")
    if away_att < 2:
        penalty = 0.08 * away_attack_scale
        away_mu_mult  -= penalty
        notes.append(f"Visitante con pocos delanteros ({away_att}), penalizacion mu -{penalty:.2f}")

    # Few defenders → opponent scores more (scaled by how defence-reliant the team is)
    if home_def < 3:
        bonus = 0.05 * away_defence_scale
        away_mu_mult  += bonus
        notes.append(f"Local con pocos defensas ({home_def}), mu visitante +{bonus:.2f}")
    if away_def < 3:
        bonus = 0.05 * home_defence_scale
        home_lam_mult += bonus
        notes.append(f"Visitante con pocos defensas ({away_def}), lam local +{bonus:.2f}")

    return {
        "home_lam_mult": round(max(0.80, min(1.20, home_lam_mult)), 3),
        "away_mu_mult":  round(max(0.80, min(1.20, away_mu_mult)),  3),
        "notes":         notes,
        "home_xi":       [p["name"] for p in home_players],
        "away_xi":       [p["name"] for p in away_players],
        "home_xg_profile": home_profile,
        "away_xg_profile": away_profile,
    }
