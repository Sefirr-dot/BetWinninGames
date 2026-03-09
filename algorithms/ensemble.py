"""
Ensemble model + Kelly Criterion ranking.

Combines Dixon-Coles, Elo, Form, BTTS, Corners, H2H and fatigue.
Ranks picks by a profitability score based on Kelly Criterion logic.
"""

from datetime import date as _date
from collections import Counter

from config import (MODEL_WEIGHTS, HIGH_CONFIDENCE_THRESHOLD, MEDIUM_CONFIDENCE_THRESHOLD,
                    DRAW_RATE_BY_LEAGUE, MARKET_BLEND_WEIGHT,
                    HIGH_PROB_CORRECTION_ALPHA, HIGH_PROB_CORRECTION_THRESHOLD)
from algorithms import dixon_coles, elo, form, btts, corners, cards, h2h, fatigue
from algorithms import simulate as _simulate
from algorithms import motivation as _motivation
from algorithms import referee as _referee
import match_context as _match_context
from algorithms import calibrator as _calibrator
from algorithms import weight_optimizer as _wopt
from algorithms import meta_learner as _meta_learner

# Load once at import time; falls back to config defaults when file absent
_calib_params    = _calibrator.load_calibrator()
_dynamic_weights = _wopt.load_weights()
_eff_weights     = {**MODEL_WEIGHTS, **_dynamic_weights} if _dynamic_weights else MODEL_WEIGHTS
_ml_model        = _meta_learner.load_model()


def _blend_1x2(
    dc_pred: dict,
    elo_pred: dict,
    form_pred: dict,
    h2h_pred: dict,
) -> tuple[float, float, float]:
    """Weighted blend of 1X2 probabilities. H2H weight only applied when sufficient."""
    w_dc  = _eff_weights["dixon_coles"]
    w_elo = _eff_weights["elo"]
    w_frm = _eff_weights["form"]
    w_h2h = _eff_weights["h2h"] if h2h_pred.get("sufficient") else 0.0
    total_w = w_dc + w_elo + w_frm + w_h2h

    def safe(d: dict, key: str) -> float:
        return d.get(key, 1 / 3) if d else 1 / 3

    ph = (
        w_dc  * safe(dc_pred,   "prob_home") +
        w_elo * safe(elo_pred,  "prob_home") +
        w_frm * safe(form_pred, "prob_home") +
        w_h2h * safe(h2h_pred,  "prob_home")
    ) / total_w

    pd = (
        w_dc  * safe(dc_pred,   "prob_draw") +
        w_elo * safe(elo_pred,  "prob_draw") +
        w_frm * safe(form_pred, "prob_draw") +
        w_h2h * safe(h2h_pred,  "prob_draw")
    ) / total_w

    pa = (
        w_dc  * safe(dc_pred,   "prob_away") +
        w_elo * safe(elo_pred,  "prob_away") +
        w_frm * safe(form_pred, "prob_away") +
        w_h2h * safe(h2h_pred,  "prob_away")
    ) / total_w

    total = ph + pd + pa
    return ph / total, pd / total, pa / total


def _xg_strength(form_dict: dict) -> float:
    """Normalized xG net strength for a team's form dict. Returns 0.01..1.0."""
    xg_for = form_dict.get("xg_scored_pg", form_dict.get("goals_scored_pg", 1.2))
    xg_ag  = form_dict.get("xg_conceded_pg", form_dict.get("goals_conceded_pg", 1.2))
    denom  = xg_for + xg_ag + 0.5
    return max(0.01, min(1.0, (xg_for - xg_ag + denom) / (2 * denom)))


def _kelly_score(prob: float, model_confidence: float, bk_odds: float | None = None) -> float:
    """
    Kelly-based profitability score.
    Uses real bookmaker odds when available; falls back to an 8% margin proxy.
    """
    if bk_odds and bk_odds > 1.0:
        implied = 1.0 / bk_odds
        edge = prob - implied
    else:
        # No real odds available: proxy with ~8% bookmaker margin on best prob
        edge = prob * 0.08
    return max(0.0, prob * model_confidence * (1 + max(0.0, edge)))


def _apply_high_prob_correction(ph: float, pd: float, pa: float) -> tuple[float, float, float]:
    """
    Correct systematic underestimation of high-probability outcomes.

    Backtest evidence (FL1 2023-24):
      60-70% predicted → 69.3% actual  (+4.7%)
      70%+   predicted → 82.9% actual  (+7.8%)

    Formula: p_corr = p + alpha * (p - threshold) * (1 - p)
    Only applies to the highest-probability outcome; redistributes from others.
    Only called when neither Platt calibrator nor meta_learner is active.
    """
    probs = [ph, pd, pa]
    best_idx = max(range(3), key=lambda i: probs[i])
    p = probs[best_idx]

    if p <= HIGH_PROB_CORRECTION_THRESHOLD:
        return ph, pd, pa

    correction = HIGH_PROB_CORRECTION_ALPHA * (p - HIGH_PROB_CORRECTION_THRESHOLD) * (1.0 - p)
    probs[best_idx] = p + correction
    total = sum(probs)
    return probs[0] / total, probs[1] / total, probs[2] / total


def _confidence(
    dc_pred: dict,
    elo_pred: dict,
    form_pred: dict,
    h2h_pred: dict,
) -> float:
    """
    Full-distribution confidence score (v2).

    Combines two signals:
      1. Total variance across all three outcome probabilities (home/draw/away)
         — captures disagreement in the entire distribution, not just the winner.
      2. Consensus rate — fraction of models that agree on WHO wins.
         Penalises splits like DC→home, Elo→draw, Form→away.

    Range: 0.55 (high disagreement) → 1.0 (perfect consensus).
    """
    models = [d for d in (dc_pred, elo_pred, form_pred) if d]
    if h2h_pred.get("sufficient"):
        models.append(h2h_pred)

    if not models:
        return 0.5

    n = len(models)

    def _winner_key(d: dict) -> str:
        p = {"H": d.get("prob_home", 0), "D": d.get("prob_draw", 0), "A": d.get("prob_away", 0)}
        return max(p, key=p.get)

    # 1. Total variance across all 3 outcome distributions
    total_var = 0.0
    for key in ("prob_home", "prob_draw", "prob_away"):
        probs  = [d.get(key, 0.0) for d in models]
        mean_p = sum(probs) / n
        total_var += sum((p - mean_p) ** 2 for p in probs) / n

    # Normalise: total_var ~0.05 is very high disagreement → spread_penalty = 1.0
    spread_penalty = min(1.0, total_var / 0.05)

    # 2. Consensus: fraction of models that predict the same winner
    winners = [_winner_key(d) for d in models]
    consensus_rate = Counter(winners).most_common(1)[0][1] / n

    # Combine: low variance + high consensus → high confidence
    conf = consensus_rate * (1.0 - 0.45 * spread_penalty)

    return max(0.55, min(1.0, conf))


def _resolve_dc_params(dc_params: dict, league_code: str | None) -> dict:
    """
    Select the right DC params dict from a per-league or plain params dict.

    dc_params can be:
      - A plain single-model dict (legacy)        → returned as-is
      - A per-league dict {"PL": {...}, "_global": {...}}  → picks league or _global
    """
    if not dc_params:
        return {}
    # Per-league dict has a "_global" key
    if "_global" in dc_params:
        if league_code and league_code in dc_params and dc_params[league_code]:
            return dc_params[league_code]
        return dc_params.get("_global", {})
    # Plain single-model dict
    return dc_params


def predict_match(
    home_id: int,
    away_id: int,
    dc_params: dict,
    elo_ratings: dict,
    all_matches: list[dict],
    reference_date: _date | None = None,
    league_code: str | None = None,
    standings_map: dict | None = None,
    market_odds: dict | None = None,
    elo_home_ratings: dict | None = None,
    elo_away_ratings: dict | None = None,
    odds_age_hours: float | None = None,
    referee_name: str | None = None,
) -> dict:
    """
    Full prediction for one match.

    Parameters
    ----------
    reference_date    : date of the match (used for fatigue calculation).
    league_code       : e.g. "PL", "PD" — selects the right per-league DC model.
    standings_map     : {team_id: position} for table-position adjustment.
    market_odds       : {odds_1, odds_x, odds_2} from bookmaker CSV.
                        When provided, final probs are blended with market
                        implied probs weighted by MARKET_BLEND_WEIGHT.
    elo_home_ratings  : venue-specific home Elo ratings from build_split_ratings().
    elo_away_ratings  : venue-specific away Elo ratings from build_split_ratings().
    """
    if reference_date is None:
        reference_date = _date.today()

    league_dc = _resolve_dc_params(dc_params, league_code)

    # --- Fatigue ---
    fat = fatigue.compute(home_id, away_id, all_matches, reference_date)

    # --- Dixon-Coles (with fatigue applied to expected goals) ---
    dc_pred = dixon_coles.predict(
        home_id, away_id, league_dc,
        home_fatigue=fat["home_mult"],
        away_fatigue=fat["away_mult"],
    )

    # --- Elo (venue-specific ratings when available) ---
    elo_pred = elo.predict(
        home_id, away_id, elo_ratings, league=league_code or "",
        home_ratings=elo_home_ratings,
        away_ratings=elo_away_ratings,
    )

    # --- Form (SoS-adjusted via Elo ratings) ---
    form_pred = form.predict(home_id, away_id, all_matches, elo_ratings=elo_ratings)

    # --- H2H ---
    h2h_pred = h2h.predict(home_id, away_id, all_matches, reference_date=reference_date)

    # --- Extract positions early (needed for context features AND position adjustment) ---
    home_pos = (standings_map or {}).get(home_id)
    away_pos = (standings_map or {}).get(away_id)

    # --- Build context features for meta-learner ---
    _hf = (form_pred or {}).get("home_form", {})
    _af = (form_pred or {}).get("away_form", {})
    _fatigue_diff = (fat.get("home_days", 7) - fat.get("away_days", 7)) / 7.0
    _fatigue_diff = max(-1.0, min(1.0, _fatigue_diff))
    _pos_diff = (away_pos - home_pos) / 19.0 if (home_pos and away_pos) else 0.0
    _pos_diff = max(-1.0, min(1.0, _pos_diff))

    # Market implied probs (margin-removed) — 1/3 fallback when no odds available
    _mkt_ph = _mkt_px = _mkt_pa = 1 / 3
    if market_odds:
        try:
            _r1 = 1.0 / market_odds["odds_1"]
            _rx = 1.0 / market_odds["odds_x"]
            _r2 = 1.0 / market_odds["odds_2"]
            _rs = _r1 + _rx + _r2
            if _rs > 0:
                _mkt_ph, _mkt_px, _mkt_pa = _r1 / _rs, _rx / _rs, _r2 / _rs
        except (KeyError, ZeroDivisionError):
            pass

    # --- Blend 1X2 (meta-learner takes priority over weighted blend) ---
    _sub_preds = {
        "dc": dc_pred, "elo": elo_pred, "form": form_pred, "h2h": h2h_pred,
        "context": {
            "xg_form_home": round(_xg_strength(_hf), 6),
            "xg_form_away": round(_xg_strength(_af), 6),
            "pos_diff_norm": round(_pos_diff, 6),
            "fatigue_diff":  round(_fatigue_diff, 6),
            "mkt_ph":       round(_mkt_ph, 6),
            "mkt_px":       round(_mkt_px, 6),
            "mkt_pa":       round(_mkt_pa, 6),
        },
    }
    ml_result  = _meta_learner.predict(_sub_preds, _ml_model)

    if ml_result is not None:
        # XGBoost outputs calibrated softmax probs — skip Platt calibration
        ph, pd, pa = ml_result
    else:
        ph, pd, pa = _blend_1x2(dc_pred, elo_pred, form_pred, h2h_pred)
        if _calib_params:
            ph, pd, pa = _calibrator.apply_calibration(ph, pd, pa, _calib_params)

    # --- Expected goals (fatigue already baked in via DC) ---
    lam = dc_pred.get("lambda_", 1.3) if dc_pred else 1.3
    mu  = dc_pred.get("mu_",      1.0) if dc_pred else 1.0

    # --- Motivation adjustment ---
    motiv = _motivation.from_standings(home_pos, away_pos, league_code or "")
    lam = round(lam * motiv["home_mult"], 4)
    mu  = round(mu  * motiv["away_mult"], 4)

    # --- Draw adjustment: nudge draw prob toward league historical rate ---
    # Enhanced draw detector: composite score amplifies nudge when multiple signals align.
    elo_diff = abs(elo_pred.get("rating_home", 1500) - elo_pred.get("rating_away", 1500))
    proximity = max(0.0, 1.0 - elo_diff / 500.0)
    league_draw_rate = DRAW_RATE_BY_LEAGUE.get(league_code or "", 0.25)

    draw_signals = 0
    if elo_diff < 100:                                            # closely-matched teams
        draw_signals += 1
    if (lam + mu) < 2.5:                                         # low-scoring expected game
        draw_signals += 1
    if h2h_pred.get("sufficient") and h2h_pred.get("prob_draw", 0) > 0.35:  # H2H draw-heavy
        draw_signals += 1
    _h_ppg = (form_pred or {}).get("home_form", {}).get("points_per_game", 1.5)
    _a_ppg = (form_pred or {}).get("away_form", {}).get("points_per_game", 1.5)
    if 0.8 <= _h_ppg <= 2.0 and 0.8 <= _a_ppg <= 2.0:          # both teams mediocre form
        draw_signals += 1
    nudge_strength = 0.25 + 0.15 * draw_signals / 4.0           # 0.25..0.40

    pd_old = pd
    pd = max(0.05, min(0.45, pd + proximity * (league_draw_rate - pd) * nudge_strength))
    actual_adj = pd - pd_old
    if abs(actual_adj) > 1e-9:
        denom = ph + pa + 1e-9
        ph = max(0.0, ph - actual_adj * ph / denom)
        pa = max(0.0, pa - actual_adj * pa / denom)
        _t = ph + pd + pa
        ph, pd, pa = ph / _t, pd / _t, pa / _t

    # --- Position adjustment: table rank as tiebreaker ---
    if home_pos and away_pos:
        pos_diff = away_pos - home_pos  # positive = home team is ranked higher
        pos_edge = max(-0.05, min(0.05, pos_diff / 20 * 0.05))
        ph += pos_edge * (1 - ph)
        pa = max(0.0, pa - pos_edge * pa)
        _t = ph + pd + pa
        ph, pd, pa = ph / _t, pd / _t, pa / _t

    # --- High-probability correction (only when raw ensemble is used) ---
    # Corrects systematic underestimation at high probs found in backtest analysis.
    # Skipped when meta_learner (already calibrated) or Platt calibrator is active.
    if ml_result is None and not _calib_params:
        ph, pd, pa = _apply_high_prob_correction(ph, pd, pa)

    # --- Market odds blend (when bookmaker CSV available for this match) ---
    # Removes bookmaker margin via normalization, then blends MARKET_BLEND_WEIGHT
    # of market consensus into the final probability. Market odds are highly
    # efficient and encode crowd wisdom not captured by statistical models.
    market_blend_applied = False
    if market_odds and MARKET_BLEND_WEIGHT > 0:
        try:
            r1 = 1.0 / market_odds["odds_1"]
            rx = 1.0 / market_odds["odds_x"]
            r2 = 1.0 / market_odds["odds_2"]
            mkt_sum = r1 + rx + r2
            if mkt_sum > 0:
                mkt_ph, mkt_pd, mkt_pa = r1 / mkt_sum, rx / mkt_sum, r2 / mkt_sum
                # Stale odds penalty: reduce blend weight when CSV is old.
                # >6h old → 25% weight; 2-6h → 50%; <2h → full weight.
                w = MARKET_BLEND_WEIGHT
                if odds_age_hours is not None:
                    if odds_age_hours > 6:
                        w *= 0.25
                        print(f"    [odds] Cuotas con {odds_age_hours:.1f}h de antigüedad — blend reducido al 25%.")
                    elif odds_age_hours > 2:
                        w *= 0.50
                ph = (1 - w) * ph + w * mkt_ph
                pd = (1 - w) * pd + w * mkt_pd
                pa = (1 - w) * pa + w * mkt_pa
                # convex combination → already sums to 1.0
                market_blend_applied = True
        except (KeyError, ZeroDivisionError):
            pass

    # --- Referee adjustment (small home/away bias correction) ---
    _ref_profiles = _referee.load_profiles()
    ref_adj = _referee.get_adjustments(referee_name, league_code or "", _ref_profiles)
    if ref_adj["known"] and abs(ref_adj["home_prob_adj"]) > 0.001:
        adj = ref_adj["home_prob_adj"]
        ph = max(0.05, ph + adj)
        pa = max(0.05, pa - adj * 0.7)   # partly absorbed from draw too
        s  = ph + pd + pa
        ph, pd, pa = ph / s, pd / s, pa / s

    # --- BTTS (Poisson exact using fatigue-adjusted lambda/mu) ---
    btts_pred = btts.predict(home_id, away_id, all_matches, lambda_=lam, mu_=mu,
                             league_code=league_code)
    btts_prob = btts_pred["btts_prob"]

    # Slight H2H correction to BTTS when we have historical data
    if h2h_pred.get("sufficient"):
        btts_prob = 0.85 * btts_prob + 0.15 * h2h_pred["btts_rate"]
        btts_prob = max(0.10, min(0.90, btts_prob))

    # --- Monte Carlo simulation (all secondary markets) ---
    dc_rho = dc_pred.get("rho", 0.05) if dc_pred else 0.05
    mc = _simulate.simulate(lam, mu, rho=dc_rho)

    # Use MC for secondary markets — more accurate than Poisson-exact and
    # adds Over 1.5 / Over 3.5 / Over 4.5 / Asian HCap / BTTS combos
    dc_over25 = mc["over25"]

    # --- Corners ---
    corners_pred = corners.predict(lam, mu)

    # --- Cards ---
    cards_pred = cards.predict(lam, mu)

    # --- Confidence & Kelly ---
    conf = _confidence(dc_pred, elo_pred, form_pred, h2h_pred)

    best_outcome = max([("home", ph), ("draw", pd), ("away", pa)], key=lambda x: x[1])
    best_prob  = best_outcome[1]
    best_label = best_outcome[0]

    # Use real bookmaker odds for the best outcome when available
    _bk_odds_best = None
    if market_odds:
        _bk_odds_best = {
            "home": market_odds.get("odds_1"),
            "draw": market_odds.get("odds_x"),
            "away": market_odds.get("odds_2"),
        }.get(best_label)

    profitability_score = _kelly_score(best_prob, conf, _bk_odds_best)

    if best_prob >= HIGH_CONFIDENCE_THRESHOLD and conf >= 0.80:
        stars = 5
    elif best_prob >= HIGH_CONFIDENCE_THRESHOLD:
        stars = 4
    elif best_prob >= MEDIUM_CONFIDENCE_THRESHOLD:
        stars = 3
    elif best_prob >= 0.45:
        stars = 2
    else:
        stars = 1

    # Full-distribution confidence penalty: models predict different winners
    # or show high spread → downgrade one star to avoid false high-confidence picks
    if conf < 0.65:
        stars = max(stars - 1, 1)

    return {
        "prob_home": ph,
        "prob_draw": pd,
        "prob_away": pa,
        "best_outcome": best_label,
        "best_prob": best_prob,
        "profitability_score": profitability_score,
        "stars": stars,
        "model_confidence": conf,
        "over25": dc_over25,
        "over35": dc_pred.get("over35", 0.30) if dc_pred else 0.30,
        "btts_prob": btts_prob,
        "expected_goals_home": lam,
        "expected_goals_away": mu,
        "most_likely_score": dc_pred.get("most_likely_score", (1, 1)) if dc_pred else (1, 1),
        "most_likely_score_prob": dc_pred.get("most_likely_score_prob", 0.10) if dc_pred else 0.10,
        "corners": corners_pred,
        "cards": cards_pred,
        "dc": dc_pred,
        "elo": elo_pred,
        "form": form_pred,
        "btts": btts_pred,
        "h2h": h2h_pred,
        "fatigue": fat,
        "home_pos": home_pos,
        "away_pos": away_pos,
        "mc": mc,
        "motivation": motiv,
        "referee_adj": ref_adj,
        "_context": _sub_preds["context"],
        "_used_meta_learner": ml_result is not None,
        "_market_blend_applied": market_blend_applied,
        "_tags": (
            _match_context.classify(
                elo_pred   = elo_pred,
                form_pred  = form_pred,
                h2h_pred   = h2h_pred,
                home_pos   = home_pos,
                away_pos   = away_pos,
            ) + motiv["tags"]
        ),
    }


def rank_predictions(predictions: list[dict]) -> list[dict]:
    """Sort predictions by profitability_score descending."""
    def _score(p):
        inner = p.get("prediction", p)
        return inner.get("profitability_score", 0.0)
    return sorted(predictions, key=_score, reverse=True)
