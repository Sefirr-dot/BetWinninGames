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
import json
import os
import re
import unicodedata

from collections import defaultdict

from config import (ODDS_DIR, VALUE_BET_EDGE_THRESHOLD, VALUE_BET_EDGE_STEP,
                    VALUE_BET_MIN_STARS, VALUE_BET_MIN_STARS_BY_LEAGUE,
                    VALUE_BET_EDGE_THRESHOLD_BY_LEAGUE,
                    VALUE_BET_MIN_ODDS,
                    ANTIDRAW_SQUEEZE_THRESHOLD, ANTIDRAW_SQUEEZE_FACTOR,
                    ANTIDRAW_EDGE_BONUS_MAX)

_TRACKER_METRICS_PATH = "cache/tracker_metrics.json"
_KELLY_MIN_PICKS = 20   # minimum resolved picks per league to trust the ROI estimate


def _load_league_kelly_multipliers() -> dict[str, float]:
    """
    Load per-league ROI from tracker_metrics.json and compute Kelly multipliers.

    multiplier = max(0.30, min(1.50, 1.0 + league_roi))
      - league ROI +10% → multiplier 1.10 (bet slightly more)
      - league ROI  -5% → multiplier 0.95 (bet slightly less)
      - league ROI -70% → multiplier 0.30 (floor — never stake zero)

    Returns an empty dict when the file doesn't exist or has insufficient data,
    which causes find_edges() to fall back to the default Kelly fraction.
    """
    try:
        with open(_TRACKER_METRICS_PATH, encoding="utf-8") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

    multipliers: dict[str, float] = {}
    per_league = data.get("per_league", {})
    for lg, m in per_league.items():
        if (m.get("n") or 0) < _KELLY_MIN_PICKS:
            continue   # not enough data — skip this league
        roi = m.get("roi") or 0.0
        multipliers[lg] = max(0.30, min(1.50, 1.0 + roi))
    return multipliers


# Loaded once at import time; refreshed each tracker.py run via file write
_LEAGUE_KELLY_MULT: dict[str, float] = _load_league_kelly_multipliers()


# Adaptive fractional Kelly multipliers by (stars, edge) tier
# Full Kelly only for highest-conviction bets; quarter Kelly as default
def _kelly_fraction(edge: float, bk_odds: float, stars: int, league: str = "") -> float:
    """
    Tiered fractional Kelly staking.

    Tier  | Stars | Edge     | Kelly mult
    ------|-------|----------|------------
    Full  | >= 5  | >= 15%   | 1.0
    Half  | >= 4  | >= 10%   | 0.5
    Qtr   | any   | any      | 0.25

    Additionally scaled by the per-league ROI multiplier loaded from
    cache/tracker_metrics.json (when >= _KELLY_MIN_PICKS resolved picks exist).
    Hard cap at 25% regardless of tier.
    """
    raw = edge * bk_odds / (bk_odds - 1)
    if stars >= 5 and edge >= 0.15:
        mult = 1.0
    elif stars >= 4 and edge >= 0.10:
        mult = 0.5
    else:
        mult = 0.25
    league_mult = _LEAGUE_KELLY_MULT.get(league, 1.0)
    return min(raw * mult * league_mult, 0.25)

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
                        "bk_1":       row.get("bk_1", ""),
                        "bk_x":       row.get("bk_x", ""),
                        "bk_2":       row.get("bk_2", ""),
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


# ---------------------------------------------------------------------------
# Market Inefficiency Score (MIS)
# ---------------------------------------------------------------------------

# Static correlation table for same-match outcome pairs.
# Based on football analytics: home win and away win are near-mutually-exclusive;
# home/away win and over 2.5 are positively correlated (decisive wins tend to be high-scoring).
_OUTCOME_CORRELATION: dict[tuple[str, str], float] = {
    ("home",   "away"):   -0.90,
    ("home",   "draw"):   -0.70,
    ("draw",   "away"):   -0.70,
    ("home",   "over25"):  0.30,
    ("away",   "over25"):  0.25,
    ("home",   "btts"):    0.15,
    ("away",   "btts"):    0.15,
    ("over25", "btts"):    0.55,
    ("draw",   "over25"): -0.20,
    ("draw",   "btts"):    0.10,
}


def _get_corr(a: str, b: str) -> float:
    """
    Look up SIGNED correlation between two outcomes (symmetric).

    Returns the signed correlation, NOT abs(). Negative correlations (e.g.
    home+away = -0.90) mean the outcomes tend not to co-occur — they act as a
    partial hedge, so no Kelly discount is warranted. The discount formula in
    _corr_discount clamps negative correlations to 0 (no discount).

    Only positive correlations (e.g. over25+btts = +0.55) warrant a Kelly reduction
    because both bets tend to win/lose together, amplifying portfolio variance.
    """
    return _OUTCOME_CORRELATION.get((a, b), _OUTCOME_CORRELATION.get((b, a), 0.0))


def _corr_discount(n_correlated: int, max_corr: float) -> float:
    """
    Kelly discount factor when multiple bets are on the same match.

    Only positive correlations increase portfolio variance and warrant a discount.
    Negative correlations (partial hedges) get no discount (clamped to 0).

    Formula: 1 / (1 + (n-1) * max(0, corr)) — derived from Thorp (1997) Kelly
    extension for correlated bets.

    Returns multiplier in [0.30, 1.0]:
      1 bet          → 1.0   (no discount)
      2, corr=0.0   → 1.0   (independent or negative — no discount)
      2, corr=+0.55 → 0.65  (over25 + btts)
      2, corr=+0.30 → 0.77  (home win + over25)
    """
    if n_correlated <= 1:
        return 1.0
    # Clamp to 0: negative correlations never warrant a Kelly reduction
    effective_corr = max(0.0, max_corr)
    if effective_corr < 0.01:
        return 1.0
    return max(0.30, 1.0 / (1.0 + (n_correlated - 1) * effective_corr))


def compute_mis(
    effective_edge: float,
    sharp_money: bool,
    odds_age_hours: float | None,
    clv_vs_pinnacle: float | None,
    bk_odds: float = 2.0,
) -> float:
    """
    Market Inefficiency Score — composite signal in [0.25, 1.50].

    Higher MIS = stronger evidence that the edge is real and the market is soft.
    Used to scale the Kelly fraction: kelly_final = kelly_base * MIS.

    Components
    ----------
    edge_magnitude  : normalised to 20% edge = full +0.25 contribution
    sharp_money     : odds shortened 10%+ on our outcome → +0.10
    stale_odds      : older than 6h → -0.15, older than 2h → -0.07
    clv_vs_pinnacle : positive CLV vs sharp closing line → up to +0.10

    Baseline of 0.50 ensures we never reduce Kelly to zero from MIS alone.
    """
    score = 0.50

    # 1. Edge magnitude (bounded: 20% edge = maximum contribution of +0.25)
    score += 0.25 * min(1.0, effective_edge / 0.20)

    # 2. Sharp money confirmation
    if sharp_money:
        score += 0.10

    # 3. Stale odds penalty (market may have moved without our data updating)
    if odds_age_hours is not None:
        if odds_age_hours > 6:
            score -= 0.15
        elif odds_age_hours > 2:
            score -= 0.07

    # 4. CLV vs Pinnacle (positive = we predicted better than the sharpest closing line)
    if clv_vs_pinnacle is not None:
        # ±10% CLV maps to ±0.10 contribution; clamped at ±0.10
        score += 0.10 * max(-1.0, min(1.0, clv_vs_pinnacle / 0.10))

    return round(max(0.25, min(1.50, score)), 3)


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
        league = mi.get("_league_code", "")
        min_stars = VALUE_BET_MIN_STARS_BY_LEAGUE.get(league, VALUE_BET_MIN_STARS)
        if stars < min_stars:
            continue
        base_threshold = VALUE_BET_EDGE_THRESHOLD_BY_LEAGUE.get(league, VALUE_BET_EDGE_THRESHOLD)
        effective_threshold = base_threshold + (5 - stars) * VALUE_BET_EDGE_STEP

        # Odds movement signal (sharp money)
        match_date = mi.get("utcDate", "")[:10]
        try:
            from odds_fetcher import get_odds_movement, get_pinnacle_implied
            movement = get_odds_movement(home_name, away_name, match_date)
            pinnacle_implied = get_pinnacle_implied(home_name, away_name, match_date)
        except Exception:
            movement = None
            pinnacle_implied = None

        # Map outcome → (model_prob, bk_odds, best_bookmaker_name)
        checks = [
            ("home",  pred.get("prob_home", 0.0), odds["odds_1"], odds.get("bk_1", "")),
            ("draw",  pred.get("prob_draw", 0.0), odds["odds_x"], odds.get("bk_x", "")),
            ("away",  pred.get("prob_away", 0.0), odds["odds_2"], odds.get("bk_2", "")),
        ]

        # Add Over 2.5 and BTTS when bookmaker odds are available in the CSV
        # (no bk name tracking for these markets — best odds across books already applied)
        if (odds.get("odds_o25") or 0) > 1.0:
            checks.append(("over25", pred.get("over25", 0.0), odds["odds_o25"], ""))
        if (odds.get("odds_btts") or 0) > 1.0:
            checks.append(("btts", pred.get("btts_prob", 0.0), odds["odds_btts"], ""))

        # Anti-draw squeeze: when market overprices draw vs model, the model is
        # confident that home/away is underpriced. Partially redistribute the
        # draw mispricing as an edge bonus on the competing outcomes.
        # Derived from draw_model weight mkt=-0.62 (market overvalues draws).
        _draw_squeeze = 0.0
        try:
            _r1 = 1 / odds["odds_1"]; _rx = 1 / odds["odds_x"]; _r2 = 1 / odds["odds_2"]
            _rs = _r1 + _rx + _r2
            _mkt_draw_clean = _rx / _rs
            _gap = _mkt_draw_clean - pred.get("prob_draw", 0.0)
            if _gap > ANTIDRAW_SQUEEZE_THRESHOLD:
                _draw_squeeze = min(ANTIDRAW_EDGE_BONUS_MAX,
                                    _gap * ANTIDRAW_SQUEEZE_FACTOR)
        except (KeyError, ZeroDivisionError):
            pass

        # Fetch odds age for MIS staleness penalty
        _odds_age = None
        try:
            from odds_fetcher import get_odds_age_hours
            _odds_age = get_odds_age_hours(home_name, away_name, match_date)
        except Exception:
            pass

        # Extract sub-model probs for sharpness agreement scoring
        _dc_sub  = pred.get("dc")  or {}
        _elo_sub = pred.get("elo") or {}

        for outcome, model_prob, bk_odds, best_book in checks:
            if bk_odds <= VALUE_BET_MIN_ODDS:
                continue  # market too efficient at short odds — skip
            implied_prob = 1.0 / bk_odds
            edge = model_prob - implied_prob

            # Sharp money bonus: odds shortened >10% on same outcome as model
            _outcome_mv_key = {"home": "home", "draw": "draw", "away": "away",
                                "over25": None, "btts": None}.get(outcome)
            mv_ratio = (movement.get(_outcome_mv_key, 1.0)
                        if movement and _outcome_mv_key else 1.0)
            sharp_money = mv_ratio >= 1.10

            # Anti-draw squeeze bonus applies only to home/away (not draw/markets)
            _squeeze_bonus = _draw_squeeze if outcome in ("home", "away") else 0.0
            effective_edge = edge + (0.02 if sharp_money else 0.0) + _squeeze_bonus

            if effective_edge >= effective_threshold:
                # CLV vs Pinnacle: positive = we predicted better than sharp closing line
                _pin_key = {"home": "home", "draw": "draw", "away": "away"}.get(outcome)
                pin_prob = (pinnacle_implied.get(_pin_key)
                            if pinnacle_implied and _pin_key else None)
                clv_vs_pinnacle = round(model_prob - pin_prob, 4) if pin_prob else None

                # Market Inefficiency Score — scales Kelly fraction
                mis = compute_mis(
                    effective_edge  = effective_edge,
                    sharp_money     = sharp_money,
                    odds_age_hours  = _odds_age,
                    clv_vs_pinnacle = clv_vs_pinnacle,
                    bk_odds         = bk_odds,
                )

                kelly_base = _kelly_fraction(effective_edge, bk_odds, stars, league)
                # Hard cap 25% is enforced AFTER MIS scaling (not inside _kelly_fraction)
                kelly = round(min(kelly_base * mis, 0.25), 5)

                # Sharpness Score — composite market intelligence signal (0-100)
                _dc_prob_out  = {
                    "home": _dc_sub.get("prob_home"),
                    "draw": _dc_sub.get("prob_draw"),
                    "away": _dc_sub.get("prob_away"),
                }.get(outcome)
                _elo_prob_out = {
                    "home": _elo_sub.get("prob_home"),
                    "draw": _elo_sub.get("prob_draw"),
                    "away": _elo_sub.get("prob_away"),
                }.get(outcome)
                try:
                    from algorithms.sharpness import compute_sharpness_score
                    _sharpness = compute_sharpness_score(
                        edge            = effective_edge,
                        sharp_money     = sharp_money,
                        clv_vs_pinnacle = clv_vs_pinnacle,
                        odds_movement   = mv_ratio if movement and _outcome_mv_key else None,
                        mis             = mis,
                        dc_prob         = _dc_prob_out,
                        elo_prob        = _elo_prob_out,
                        model_prob      = model_prob,
                    )
                except Exception:
                    _sharpness = 0

                value_bets.append({
                    "match":           f"{home_name} vs {away_name}",
                    "league":          mi.get("_league_code", ""),
                    "outcome":         outcome,
                    "model_prob":      model_prob,
                    "implied_prob":    implied_prob,
                    "edge":            effective_edge,
                    "bk_odds":         bk_odds,
                    "best_book":       best_book,
                    "kelly_fraction":  kelly,
                    "kelly_base":      kelly_base,
                    "market_inefficiency_score": mis,
                    "home_name":       home_name,
                    "away_name":       away_name,
                    "odds_movement":   mv_ratio if movement and _outcome_mv_key else None,
                    "sharp_money":       sharp_money,
                    "antidraw_squeeze":  round(_squeeze_bonus * 100, 1) if _squeeze_bonus else None,
                    "pinnacle_prob":     pin_prob,
                    "clv_vs_pinnacle":   clv_vs_pinnacle,
                    "corr_discount":     1.0,  # filled by post-processing below
                    "sharpness_score":   _sharpness,
                    "match_date":        match_date,
                })

    # ── Post-process: apply correlation discounts to same-match bet groups ──
    match_groups: dict[tuple, list] = defaultdict(list)
    for vb in value_bets:
        match_groups[(vb["home_name"], vb["away_name"])].append(vb)

    for group in match_groups.values():
        if len(group) < 2:
            continue
        # Find the maximum pairwise correlation in this group
        max_corr = 0.0
        for i, a in enumerate(group):
            for b in group[i + 1:]:
                max_corr = max(max_corr, _get_corr(a["outcome"], b["outcome"]))
        disc = _corr_discount(len(group), max_corr)
        for vb in group:
            vb["kelly_fraction"] = round(vb["kelly_fraction"] * disc, 5)
            vb["corr_discount"]  = round(disc, 3)

    return sorted(value_bets, key=lambda x: x["edge"], reverse=True)
