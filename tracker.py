"""
Tracker — result resolution, P&L metrics, and JS generation.

Usage
-----
    python tracker.py                  # update results + regenerate JS
    python tracker.py --no-update      # skip API calls, regenerate JS only
    python tracker.py --no-report      # update results but skip JS generation

Called automatically by main.py at the end of each prediction run.
"""

import argparse
import json
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
from datetime import date, datetime

import db_picks
import fetcher
from config import (CALIBRATION_MIN_SAMPLES, PICKS_DB, TRACKER_JS_PATH,
                    PSI_ALERT_THRESHOLD, PSI_LOOKBACK_RECENT,
                    SUBMODEL_ACCURACY_WINDOW)

RESULTS_JS_PATH = "visualizador/data/results.js"


# ---------------------------------------------------------------------------
# Result resolution
# ---------------------------------------------------------------------------

def _parse_actual(match_data: dict) -> tuple[str, int, int] | None:
    """
    Extract (actual_result, actual_over25, actual_btts) from a finished match.
    Returns None if the match is not yet FINISHED.
    """
    status = match_data.get("status", "")
    if status != "FINISHED":
        return None

    score = match_data.get("score", {})
    ft    = score.get("fullTime", {})
    home_goals = ft.get("home")
    away_goals = ft.get("away")

    if home_goals is None or away_goals is None:
        return None

    if home_goals > away_goals:
        result = "home"
    elif away_goals > home_goals:
        result = "away"
    else:
        result = "draw"

    actual_over25 = 1 if (home_goals + away_goals) > 2.5 else 0
    actual_btts   = 1 if home_goals > 0 and away_goals > 0 else 0

    return result, actual_over25, actual_btts


def _try_update_clv(pick: dict, match_id: int, db_path: str) -> None:
    """
    Attempt to populate closing_odds and clv for a newly-resolved pick by reading
    the Pinnacle closing line from cache/pinnacle/YYYY-MM-DD.csv.

    CLV = our_implied_prob (at prediction time) - 1/pinnacle_closing_odds.
    Positive CLV = we predicted better than the sharp closing line.

    This is required for the CLV-objective weight optimizer to have data to work with.
    Silently skips when Pinnacle data is unavailable (no API hit required).
    """
    try:
        import csv as _csv
        import unicodedata as _ud
        import re as _re

        match_date = pick.get("match_date", "")
        home       = pick.get("home_team",  "")
        away       = pick.get("away_team",  "")
        if not (match_date and home and away):
            return

        pin_path = f"cache/pinnacle/{match_date}.csv"
        if not os.path.exists(pin_path):
            return

        def _norm(n: str) -> str:
            n = n.lower().strip().replace("-", " ")
            n = _ud.normalize("NFKD", n).encode("ascii", "ignore").decode("ascii")
            n = _re.sub(r"[^a-z0-9 ]", "", n)
            return _re.sub(r"\s+", " ", n).strip()

        norm_h, norm_a = _norm(home), _norm(away)
        with open(pin_path, newline="", encoding="utf-8") as f:
            for row in _csv.DictReader(f):
                rh = _norm(row.get("home_team", ""))
                ra = _norm(row.get("away_team", ""))
                if not ((norm_h in rh or rh in norm_h) and (norm_a in ra or ra in norm_a)):
                    continue
                # Use the best_outcome to select the relevant closing odd
                outcome = pick.get("best_outcome", "")
                col = {"home": "pin_1", "draw": "pin_x", "away": "pin_2"}.get(outcome)
                if not col:
                    return
                closing_val = float(row.get(col, 0) or 0)
                if closing_val > 1.0:
                    db_picks.update_clv(match_id, closing_val, db_path)
                return
    except Exception:
        pass  # non-fatal — CLV data is supplementary


def update_results(db_path: str = PICKS_DB, quiet: bool = False) -> int:
    """
    Fetch results for all unresolved picks whose match_date <= today.
    Returns the count of newly resolved picks.
    """
    today = date.today().isoformat()
    unresolved = db_picks.get_unresolved(before_date=today, db_path=db_path)

    if not unresolved:
        if not quiet:
            print("  [tracker] No hay picks pendientes de resolver.")
        return 0

    if not quiet:
        print(f"  [tracker] Resolviendo {len(unresolved)} pick(s) pendiente(s)...")

    resolved_count = 0
    for pick in unresolved:
        match_id = pick["match_id"]
        try:
            data   = fetcher.get_match(match_id)
            parsed = _parse_actual(data)
            if parsed is None:
                continue  # not finished yet
            result, over25, btts = parsed
            db_picks.update_result(match_id, result, over25, btts, db_path)
            resolved_count += 1
            if not quiet:
                correct = "✓" if result == pick["best_outcome"] else "✗"
                print(f"    {correct} [{match_id}] {pick['home_team']} vs {pick['away_team']} "
                      f"-> {result} (pred: {pick['best_outcome']})")

            # CLV update: try to read Pinnacle closing odds from cached CSV
            # This populates closing_odds + clv columns required by weight_optimizer CLV objective.
            _try_update_clv(pick, match_id, db_path)

        except Exception as exc:
            if not quiet:
                print(f"    [WARNING] No se pudo resolver match {match_id}: {exc}")

    if not quiet:
        print(f"  [tracker] {resolved_count} pick(s) resuelto(s).")
    return resolved_count


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------

def _group_metrics(picks: list[dict]) -> dict:
    """
    Compute summary metrics for a subset of resolved picks (for per-league/stars breakdowns).
    """
    n = len(picks)
    if n == 0:
        return {"n": 0, "accuracy": None, "roi": None, "brier": None,
                "accuracy_over25": None, "accuracy_btts": None}

    correct = sum(1 for p in picks if p["best_outcome"] == p["actual_result"])

    brier_sum = 0.0
    for p in picks:
        ph  = p.get("prob_home") or 0.0
        pd_ = p.get("prob_draw") or 0.0
        pa  = p.get("prob_away") or 0.0
        tot = ph + pd_ + pa or 1.0
        ph, pd_, pa = ph / tot, pd_ / tot, pa / tot
        out = p["actual_result"]
        brier_sum += (ph - (1 if out == "home" else 0)) ** 2
        brier_sum += (pd_ - (1 if out == "draw" else 0)) ** 2
        brier_sum += (pa  - (1 if out == "away" else 0)) ** 2

    picks_with_odds = [p for p in picks if (p.get("fair_odds") or 0) > 1.0]
    roi_num = sum(
        (p["fair_odds"] - 1.0) if p["best_outcome"] == p["actual_result"] else -1.0
        for p in picks_with_odds
    )
    roi = roi_num / len(picks_with_odds) if picks_with_odds else None

    o25 = [p for p in picks if p.get("actual_over25") is not None]
    acc_o25 = (
        sum(1 for p in o25 if bool(p["actual_over25"]) == ((p.get("over25") or 0) >= 0.50)) / len(o25)
        if o25 else None
    )

    btts = [p for p in picks if p.get("actual_btts") is not None]
    acc_btts = (
        sum(1 for p in btts if bool(p["actual_btts"]) == ((p.get("btts") or 0) >= 0.50)) / len(btts)
        if btts else None
    )

    return {
        "n":               n,
        "accuracy":        round(correct / n, 4),
        "roi":             round(roi, 4) if roi is not None else None,
        "brier":           round(brier_sum / n, 4),
        "accuracy_over25": round(acc_o25, 4) if acc_o25 is not None else None,
        "accuracy_btts":   round(acc_btts, 4) if acc_btts is not None else None,
    }


def compute_metrics(resolved_picks: list[dict]) -> dict:
    """
    Compute P&L and calibration metrics from resolved picks.

    Returns a dict with:
        n_resolved, n_pending, n_total
        accuracy_1x2     (fraction correct)
        brier_score      (lower is better; naive baseline ≈ 0.667)
        roi_flat         (total profit / N using fair_odds)
        roi_kelly        (profit / N weighted by Kelly; None if no market odds)
        accuracy_over25, accuracy_btts
        bankroll_history  list of {date, bankroll} starting at 1.0
        calibration_ready bool
        n_for_calibration int
        per_league       {PL: {n, accuracy, roi, brier, accuracy_over25, accuracy_btts}, ...}
        per_stars        {"3": {...}, "4": {...}, "5": {...}}
        per_market       {1x2: {accuracy, roi}, over25: {accuracy}, btts: {accuracy}}
        hindsight_edge_by_league  {PL: float, ...}   (avg edge when market odds available)
        hindsight_edge_by_stars   {"3": float, ...}
    """
    n = len(resolved_picks)
    if n == 0:
        return {
            "n_resolved": 0, "n_pending": 0, "n_total": 0,
            "accuracy_1x2": None, "brier_score": None,
            "roi_flat": None, "roi_kelly": None,
            "accuracy_over25": None, "accuracy_btts": None,
            "bankroll_history": [],
            "calibration_ready": False, "n_for_calibration": 0,
        }

    # Sort by match date for bankroll curve
    picks_sorted = sorted(resolved_picks, key=lambda p: p["match_date"])

    correct_1x2 = 0
    brier_sum   = 0.0
    profit_flat = 0.0
    profit_kelly_sum = 0.0
    kelly_denom = 0
    correct_o25 = 0
    total_o25   = 0
    correct_btts_count = 0
    total_btts  = 0
    bankroll    = 1.0
    bankroll_history: list[dict] = []

    # Proportional stake: 1/N of initial bank so curve stays near 1.0
    # and final value = 1.0 + roi_flat (the curve is directly interpretable)
    n_bets_with_odds = sum(1 for p in picks_sorted if (p.get("fair_odds") or 0) > 1.0)
    unit_stake = 1.0 / max(n_bets_with_odds, 1)

    for pick in picks_sorted:
        outcome = pick["actual_result"]

        # 1X2 accuracy
        if pick["best_outcome"] == outcome:
            correct_1x2 += 1

        # Brier score
        ph = pick.get("prob_home") or 0.0
        pd_ = pick.get("prob_draw") or 0.0
        pa = pick.get("prob_away") or 0.0
        # Normalise in case of rounding errors
        total = ph + pd_ + pa or 1.0
        ph, pd_, pa = ph / total, pd_ / total, pa / total
        i_home  = 1 if outcome == "home"  else 0
        i_draw  = 1 if outcome == "draw"  else 0
        i_away  = 1 if outcome == "away"  else 0
        brier_sum += (ph - i_home)**2 + (pd_ - i_draw)**2 + (pa - i_away)**2

        # ROI flat (proportional stake on best_outcome at fair_odds)
        # profit_flat accumulates net units; roi_flat = profit_flat / n (fraction, not %)
        # bankroll uses unit_stake so the curve stays near 1.0
        fair_odds = pick.get("fair_odds") or 0.0
        if fair_odds > 1.0:
            if pick["best_outcome"] == outcome:
                profit_flat += fair_odds - 1.0
                bankroll    += unit_stake * (fair_odds - 1.0)
            else:
                profit_flat -= 1.0
                bankroll    -= unit_stake
        bankroll_history.append({"date": pick["match_date"], "bankroll": round(bankroll, 4)})

        # ROI Kelly (only when market_odds available)
        mkt = pick.get("market_odds") or 0.0
        best_p = pick.get("best_prob") or 0.0
        if mkt > 1.0 and best_p > 0:
            edge   = best_p - (1.0 / mkt)
            kelly  = max(0.0, min(0.25, edge / (mkt - 1.0)))
            if kelly > 0:
                if pick["best_outcome"] == outcome:
                    profit_kelly_sum += kelly * (mkt - 1.0)
                else:
                    profit_kelly_sum -= kelly
                kelly_denom += 1

        # Over 2.5
        if pick.get("actual_over25") is not None:
            total_o25 += 1
            pred_o25   = (pick.get("over25") or 0) >= 0.50
            if bool(pick["actual_over25"]) == pred_o25:
                correct_o25 += 1

        # BTTS
        if pick.get("actual_btts") is not None:
            total_btts += 1
            pred_btts   = (pick.get("btts") or 0) >= 0.50
            if bool(pick["actual_btts"]) == pred_btts:
                correct_btts_count += 1

    # Value bet specific metrics (market_odds populated = VB was flagged for best_outcome)
    vb_picks = [p for p in picks_sorted if (p.get("market_odds") or 0) > 1.0]
    vb_n = len(vb_picks)
    vb_correct = sum(1 for p in vb_picks if p["best_outcome"] == p["actual_result"])
    vb_roi_sum = 0.0
    for p in vb_picks:
        mkt = p.get("market_odds") or 0.0
        if mkt > 1.0:
            if p["best_outcome"] == p["actual_result"]:
                vb_roi_sum += mkt - 1.0
            else:
                vb_roi_sum -= 1.0

    # ── Per-league breakdown ────────────────────────────────────────────────
    leagues = sorted(set(p["league"] for p in picks_sorted if p.get("league")))
    per_league = {
        lg: _group_metrics([p for p in picks_sorted if p.get("league") == lg])
        for lg in leagues
    }

    # ── Per-stars breakdown ─────────────────────────────────────────────────
    all_stars = sorted(set(p["stars"] for p in picks_sorted if p.get("stars")))
    per_stars = {
        str(s): _group_metrics([p for p in picks_sorted if p.get("stars") == s])
        for s in all_stars
    }

    # ── Per-market breakdown ────────────────────────────────────────────────
    per_market = {
        "1x2": {
            "accuracy": round(correct_1x2 / n, 4),
            "roi":      round(profit_flat / n, 4),
        },
        "over25": {
            "accuracy": round(correct_o25 / total_o25, 4) if total_o25 else None,
        },
        "btts": {
            "accuracy": round(correct_btts_count / total_btts, 4) if total_btts else None,
        },
    }

    # ── Hindsight edge (Fase 1.4) ───────────────────────────────────────────
    # For picks where market_odds is known: hindsight_edge = outcome(1/0) - 1/market_odds
    he_by_league: dict[str, list[float]] = {}
    he_by_stars:  dict[str, list[float]] = {}
    for p in picks_sorted:
        mkt = p.get("market_odds") or 0.0
        if mkt <= 1.0:
            continue
        won  = p["best_outcome"] == p["actual_result"]
        edge = (1.0 if won else 0.0) - (1.0 / mkt)
        lg = p.get("league")
        if lg:
            he_by_league.setdefault(lg, []).append(edge)
        s = str(p.get("stars"))
        if s:
            he_by_stars.setdefault(s, []).append(edge)

    hindsight_edge_by_league = {
        lg: round(sum(v) / len(v), 4) for lg, v in he_by_league.items() if v
    }
    hindsight_edge_by_stars = {
        s: round(sum(v) / len(v), 4) for s, v in he_by_stars.items() if v
    }

    # ── CLV (Closing Line Value) ────────────────────────────────────────────
    clv_picks = [p for p in picks_sorted if p.get("clv") is not None]
    avg_clv = round(sum(p["clv"] for p in clv_picks) / len(clv_picks), 4) if clv_picks else None
    clv_by_league: dict[str, list] = {}
    for p in clv_picks:
        clv_by_league.setdefault(p.get("league", ""), []).append(p["clv"])
    avg_clv_by_league = {
        lg: round(sum(v) / len(v), 4) for lg, v in clv_by_league.items() if v
    }

    # ── Per-tag breakdown ───────────────────────────────────────────────────
    import json as _json
    tag_buckets: dict[str, list] = {}
    for p in picks_sorted:
        raw_tags = p.get("match_tags")
        try:
            tags = _json.loads(raw_tags) if raw_tags else []
        except Exception:
            tags = []
        for tag in tags:
            tag_buckets.setdefault(tag, []).append(p)
    per_tag = {tag: _group_metrics(picks) for tag, picks in tag_buckets.items() if picks}

    return {
        "n_resolved": n,
        "n_pending": 0,      # filled by caller
        "n_total": n,        # filled by caller
        "accuracy_1x2": round(correct_1x2 / n, 4),
        "brier_score": round(brier_sum / n, 4),
        "roi_flat": round(profit_flat / n, 6),
        "roi_kelly": round(profit_kelly_sum / kelly_denom, 6) if kelly_denom else None,
        "accuracy_over25": round(correct_o25 / total_o25, 4) if total_o25 else None,
        "accuracy_btts": round(correct_btts_count / total_btts, 4) if total_btts else None,
        "bankroll_history": bankroll_history,
        "calibration_ready": n >= CALIBRATION_MIN_SAMPLES,
        "n_for_calibration": n,
        "vb_n": vb_n,
        "vb_accuracy": round(vb_correct / vb_n, 4) if vb_n else None,
        "vb_roi_flat": round(vb_roi_sum / vb_n, 4) if vb_n else None,
        "per_league":  per_league,
        "per_stars":   per_stars,
        "per_market":  per_market,
        "per_tag":             per_tag,
        "avg_clv":             avg_clv,
        "avg_clv_by_league":   avg_clv_by_league,
        "n_clv_picks":         len(clv_picks),
        "hindsight_edge_by_league": hindsight_edge_by_league,
        "hindsight_edge_by_stars":  hindsight_edge_by_stars,
    }


# ---------------------------------------------------------------------------
# JS generation
# ---------------------------------------------------------------------------

def _pick_to_js(pick: dict) -> dict:
    """Convert a DB row dict to the JS object format consumed by the visualiser."""
    sub_raw = pick.get("sub_preds")
    sub_preds = None
    if sub_raw:
        try:
            parsed = json.loads(sub_raw)
            sub_preds = {
                k: {
                    "ph": round(v["prob_home"] * 100, 1),
                    "pd": round(v["prob_draw"] * 100, 1),
                    "pa": round(v["prob_away"] * 100, 1),
                }
                for k, v in parsed.items()
                if all(key in v for key in ("prob_home", "prob_draw", "prob_away"))
            }
        except Exception:
            pass

    return {
        "matchId":      pick["match_id"],
        "runDate":      pick["run_date"],
        "date":         pick["match_date"],
        "home":         pick["home_team"],
        "away":         pick["away_team"],
        "league":       pick["league"],
        "stars":        pick["stars"],
        "bestOutcome":  pick["best_outcome"],
        "bestProb":     round((pick["best_prob"] or 0) * 100, 1),
        "probHome":     round((pick["prob_home"] or 0) * 100, 1),
        "probDraw":     round((pick["prob_draw"] or 0) * 100, 1),
        "probAway":     round((pick["prob_away"] or 0) * 100, 1),
        "over25":       round((pick["over25"] or 0) * 100, 1),
        "btts":         round((pick["btts"] or 0) * 100, 1),
        "fairOdds":     pick["fair_odds"],
        "marketOdds":   pick["market_odds"],
        "actualResult": pick["actual_result"],
        "actualOver25": pick["actual_over25"],
        "actualBtts":   pick["actual_btts"],
        "subPreds":     sub_preds,
    }


def generate_tracker_js(
    all_picks: list[dict],
    metrics: dict | None,
    output_path: str = TRACKER_JS_PATH,
) -> str:
    """
    Write visualizador/data/tracker_data.js and return its path.

    Exports two JS variables:
        var TRACKER_PICKS   = [...];
        var TRACKER_METRICS = {...} | null;
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    picks_js   = [_pick_to_js(p) for p in all_picks]
    now_str    = datetime.now().strftime("%Y-%m-%d %H:%M")

    js = (
        f"// Auto-generated by BetWinninGames tracker — {now_str}\n"
        f"var TRACKER_PICKS = {json.dumps(picks_js, ensure_ascii=False, indent=2)};\n\n"
        f"var TRACKER_METRICS = {json.dumps(metrics, ensure_ascii=False, indent=2)};\n"
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(js)

    return output_path


# ---------------------------------------------------------------------------
# Calibrator hook
# ---------------------------------------------------------------------------

def maybe_fit_calibrator(resolved_picks: list[dict], db_path: str = PICKS_DB) -> None:
    """Fit and save Platt calibrator when ≥ CALIBRATION_MIN_SAMPLES picks exist."""
    if len(resolved_picks) < CALIBRATION_MIN_SAMPLES:
        return
    try:
        from algorithms import calibrator
        params = calibrator.fit_calibrator(resolved_picks)
        calibrator.save_calibrator(params)
        print(f"  [tracker] Calibrador actualizado ({len(resolved_picks)} picks resueltos).")
    except Exception as exc:
        print(f"  [tracker] Warning: no se pudo actualizar el calibrador: {exc}")


def maybe_train_meta_learner(resolved_picks: list[dict], db_path: str = PICKS_DB) -> None:
    """Train and save XGBoost meta-learner when ≥ 200 resolved picks with sub_preds exist."""
    from algorithms.meta_learner import _MIN_SAMPLES
    usable = sum(1 for p in resolved_picks if p.get("sub_preds") and p.get("actual_result"))
    if usable < _MIN_SAMPLES:
        return
    try:
        from algorithms import meta_learner
        result = meta_learner.train(db_path, real_only=True)  # nunca entrenar con seeds de backtest
        if "error" in result:
            print(f"  [tracker] Meta-learner: {result['error']}")
        else:
            print(f"  [tracker] Meta-learner XGBoost actualizado "
                  f"(n={result['n_samples']} picks reales, "
                  f"Brier val={result.get('brier_val', '?')}).")
    except Exception as exc:
        print(f"  [tracker] Warning: no se pudo entrenar el meta-learner: {exc}")


def maybe_optimize_weights(resolved_picks: list[dict]) -> None:
    """Optimise ensemble weights when ≥ WEIGHT_OPTIMIZER_MIN_SAMPLES picks exist."""
    from config import WEIGHT_OPTIMIZER_MIN_SAMPLES
    if len(resolved_picks) < WEIGHT_OPTIMIZER_MIN_SAMPLES:
        return
    try:
        from algorithms import weight_optimizer
        weights = weight_optimizer.optimize_weights(resolved_picks)
        if weights:
            weight_optimizer.save_weights(weights)
            n = weights.get("_optimised_from_n", len(resolved_picks))
            print(
                f"  [tracker] Pesos optimizados (n={n}): "
                f"DC={weights['dixon_coles']:.3f} "
                f"Elo={weights['elo']:.3f} "
                f"Form={weights['form']:.3f}"
            )
    except Exception as exc:
        print(f"  [tracker] Warning: no se pudieron optimizar los pesos: {exc}")


# ---------------------------------------------------------------------------
# Metrics JSON snapshot (for dynamic Kelly and other consumers)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Model drift detection (PSI — Population Stability Index)
# ---------------------------------------------------------------------------

def compute_psi(resolved_picks: list[dict], n_recent: int = PSI_LOOKBACK_RECENT) -> dict:
    """
    Detect model distribution shift using the two-sample Kolmogorov-Smirnov test.

    The KS test is used instead of PSI because PSI requires histogram binning and
    needs large samples per bin for reliable estimates. With n_recent=20 picks and
    7 bins, PSI has a false-alarm rate of ~42%. The KS test has better statistical
    power for small samples, requires no binning, and produces interpretable p-values.

    Drift is flagged when KS p-value < 0.05 on any of prob_home, prob_draw, prob_away.
    The "psi_max" field is kept in the return dict for frontend compatibility but now
    holds the maximum KS statistic (0-1 range; higher = more drift).

    Returns dict with ks_stat per probability, max_ks, p-values, and drift flag.
    """
    from scipy.stats import ks_2samp

    picks_sorted = sorted(resolved_picks, key=lambda p: p["match_date"])
    if len(picks_sorted) < n_recent + 10:
        return {"psi_max": None, "drift_detected": False,
                "ks_home": None, "ks_draw": None, "ks_away": None,
                "p_home":  None, "p_draw":  None, "p_away":  None,
                "n_recent": 0, "n_baseline": 0}

    recent   = picks_sorted[-n_recent:]
    baseline = picks_sorted[:-n_recent]

    def _ks_one(key: str) -> tuple[float, float]:
        base_vals   = [p.get(key) or 0.33 for p in baseline]
        recent_vals = [p.get(key) or 0.33 for p in recent]
        stat, pval  = ks_2samp(base_vals, recent_vals)
        return float(stat), float(pval)

    ks_home, p_home = _ks_one("prob_home")
    ks_draw, p_draw = _ks_one("prob_draw")
    ks_away, p_away = _ks_one("prob_away")
    max_ks          = max(ks_home, ks_draw, ks_away)
    # Drift: any probability significantly different at α=0.05 Bonferroni-corrected for 3 tests
    # → threshold = 0.05 / 3 = 0.0167
    _BONFERRONI_THRESHOLD = 0.05 / 3   # 0.0167 — corrects for 3 simultaneous tests
    drift_detected = min(p_home, p_draw, p_away) < _BONFERRONI_THRESHOLD

    return {
        "psi_max":        round(max_ks,  4),   # KS statistic, kept for frontend compat
        "ks_home":        round(ks_home, 4),
        "ks_draw":        round(ks_draw, 4),
        "ks_away":        round(ks_away, 4),
        "p_home":         round(p_home,  4),
        "p_draw":         round(p_draw,  4),
        "p_away":         round(p_away,  4),
        "drift_detected": drift_detected,
        "n_recent":       len(recent),
        "n_baseline":     len(baseline),
    }


def _alert_psi_drift(psi: dict) -> None:
    """Send Telegram warning when KS drift is detected."""
    # Note: uses ks_home/draw/away keys (KS statistic), not psi_home/draw/away
    msg = (
        f"[ALERTA] Drift del modelo detectado (KS={psi['psi_max']:.3f})\n"
        f"KS home={psi.get('ks_home', '?')} (p={psi.get('p_home', '?')}) | "
        f"KS draw={psi.get('ks_draw', '?')} (p={psi.get('p_draw', '?')}) | "
        f"KS away={psi.get('ks_away', '?')} (p={psi.get('p_away', '?')})\n"
        f"Ultimos {psi['n_recent']} picks vs {psi['n_baseline']} picks anteriores.\n"
        f"Revisar: meta_learner.pkl o cambio de temporada?"
    )
    print(f"  [tracker] {msg}")
    try:
        from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
        import requests as _req
        if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
            _req.post(
                f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
                json={"chat_id": TELEGRAM_CHAT_ID, "text": msg},
                timeout=10,
            )
    except Exception:
        pass  # non-fatal


# ---------------------------------------------------------------------------
# Bet timing analysis (CLV by hours-to-kickoff window)
# ---------------------------------------------------------------------------

def compute_timing_analysis(
    resolved_picks: list[dict],
    history_db: str = "cache/odds_history.db",
) -> dict:
    """
    Join resolved picks with odds_history to find the timing window
    (hours before kickoff) that yielded the best average CLV per league.

    Requires odds_history.db rows to have hours_to_kickoff populated
    (available after the odds_fetcher update in this release).

    Returns dict like:
        {
          "PL": {
            "window_best_clv": "24-48h",
            "windows": {
              "0-6h":   {"n": 5, "avg_clv": 0.012},
              "6-24h":  {"n": 8, "avg_clv": 0.023},
              "24-48h": {"n": 6, "avg_clv": 0.031},
              "48h+":   {"n": 3, "avg_clv": 0.018},
            }
          }, ...
        }
    """
    if not os.path.exists(history_db):
        return {}

    clv_picks = [p for p in resolved_picks if p.get("clv") is not None]
    if not clv_picks:
        return {}

    try:
        import sqlite3 as _sql
        with _sql.connect(history_db) as conn:
            conn.row_factory = _sql.Row
            timing_rows = conn.execute(
                "SELECT match_key, MIN(hours_to_kickoff) as earliest_hours "
                "FROM odds_history "
                "WHERE hours_to_kickoff IS NOT NULL "
                "GROUP BY match_key"
            ).fetchall()
    except Exception:
        return {}

    timing_map = {r["match_key"]: r["earliest_hours"] for r in timing_rows}

    _WINDOWS = [("0-6h", 0, 6), ("6-24h", 6, 24), ("24-48h", 24, 48), ("48h+", 48, 9999)]

    buckets: dict[str, dict[str, list]] = {}
    for pick in clv_picks:
        mk = f"{pick['home_team']}|{pick['away_team']}|{pick['match_date']}"
        hrs = timing_map.get(mk)
        if hrs is None or hrs < 0:
            continue  # no timing data or fetched after kickoff
        lg = pick.get("league", "ALL")
        for win_name, lo, hi in _WINDOWS:
            if lo <= hrs < hi:
                buckets.setdefault(lg, {}).setdefault(win_name, []).append(pick["clv"])
                break

    output: dict = {}
    for lg, windows in buckets.items():
        lg_out: dict = {}
        best_clv, best_win = -999.0, None
        for win_name, clv_list in windows.items():
            avg = sum(clv_list) / len(clv_list)
            lg_out[win_name] = {"n": len(clv_list), "avg_clv": round(avg, 4)}
            if avg > best_clv:
                best_clv, best_win = avg, win_name
        lg_out["window_best_clv"] = best_win
        output[lg] = lg_out

    return output


# ---------------------------------------------------------------------------
# Sub-model accuracy (rolling window)
# ---------------------------------------------------------------------------

def compute_submodel_accuracy(
    resolved_picks: list[dict],
    n_recent: int = SUBMODEL_ACCURACY_WINDOW,
) -> dict:
    """
    Compute per-sub-model 1X2 accuracy and Brier score on the most recent
    n_recent resolved picks.

    Sub-models included: dc, elo, h2h (btts has no 1X2 prediction).

    Returns:
        {
          "dc":  {"n": int, "accuracy": float, "brier": float},
          "elo": {"n": int, "accuracy": float, "brier": float},
          "h2h": {"n": int, "accuracy": float, "brier": float},
          "n_window": int,
        }
    """
    picks_sorted = sorted(resolved_picks, key=lambda p: p["match_date"])
    window = picks_sorted[-n_recent:] if len(picks_sorted) >= n_recent else picks_sorted

    stats: dict = {}

    for model_key in ("dc", "elo", "h2h"):
        usable = []
        for p in window:
            raw = p.get("sub_preds")
            if not raw:
                continue
            try:
                sub = json.loads(raw)
            except Exception:
                continue
            m = sub.get(model_key)
            if not m:
                continue
            # H2H: skip when model itself flagged insufficient data
            if model_key == "h2h" and not m.get("sufficient", True):
                continue
            actual = p.get("actual_result")
            if actual not in ("home", "draw", "away"):
                continue
            usable.append((p, m))

        if not usable:
            continue

        correct = 0
        brier_sum = 0.0
        for p, m in usable:
            ph = m.get("prob_home", 1/3)
            pd = m.get("prob_draw", 1/3)
            pa = m.get("prob_away", 1/3)
            tot = ph + pd + pa or 1.0
            ph, pd, pa = ph / tot, pd / tot, pa / tot
            best = "home" if ph >= pd and ph >= pa else ("away" if pa >= pd else "draw")
            actual = p["actual_result"]
            if best == actual:
                correct += 1
            ih = 1 if actual == "home" else 0
            id_ = 1 if actual == "draw" else 0
            ia  = 1 if actual == "away" else 0
            brier_sum += (ph - ih) ** 2 + (pd - id_) ** 2 + (pa - ia) ** 2

        n = len(usable)
        stats[model_key] = {
            "n":        n,
            "accuracy": round(correct / n, 4),
            "brier":    round(brier_sum / n, 4),
        }

    stats["n_window"] = len(window)
    return stats


def _save_results_js(all_picks: list[dict], output_path: str = RESULTS_JS_PATH) -> None:
    """
    Write visualizador/data/results.js — resolved match results for the
    frontend bet tracker to auto-settle open bets.

    Only exports live picks (source='live') that have an actual_result.
    Keyed by "{home_team}|{away_team}|{match_date}" for O(1) lookup in JS.

    Exports:
        var RESOLVED_RESULTS = { "Arsenal|Chelsea|2026-03-08": {
            result: "H",   // home wins
            over25: true,
            btts:   false,
            league: "PL",
            matchDate: "2026-03-08"
        }, ... };
    """
    _result_map = {"home": "H", "draw": "D", "away": "A"}
    resolved = {
        f"{p['home_team']}|{p['away_team']}|{p['match_date']}": {
            "result":    _result_map.get(p["actual_result"], p["actual_result"]),  # "H" | "D" | "A"
            "over25":    bool(p.get("actual_over25")),
            "btts":      bool(p.get("actual_btts")),
            "league":    p.get("league", ""),
            "matchDate": p.get("match_date", ""),
        }
        for p in all_picks
        if p.get("source") == "live" and p.get("actual_result")
    }
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    js = (
        f"// Auto-generated by BetWinninGames tracker — {now_str}\n"
        f"var RESOLVED_RESULTS = {json.dumps(resolved, ensure_ascii=False, indent=2)};\n"
    )
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(js)


_METRICS_JSON_PATH = "cache/tracker_metrics.json"

def _save_metrics_json(metrics: dict) -> None:
    """
    Persist a lightweight snapshot of tracker metrics to cache/tracker_metrics.json.
    Used by value_detector.py (dynamic Kelly), and the frontend TRACK view.
    Only saves the fields that downstream consumers need (avoids serialising
    the full bankroll_history list which can be large).
    """
    snapshot = {
        "n_resolved":           metrics.get("n_resolved", 0),
        "accuracy_1x2":         metrics.get("accuracy_1x2"),
        "roi_flat":             metrics.get("roi_flat"),
        "per_league":           metrics.get("per_league", {}),
        "per_stars":            metrics.get("per_stars", {}),
        # New fields
        "psi":                  metrics.get("psi", {}),
        "timing_analysis":      metrics.get("timing_analysis", {}),
        "submodel_accuracy_30": metrics.get("submodel_accuracy_30", {}),
    }
    try:
        with open(_METRICS_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(snapshot, f, indent=2)
    except Exception as exc:
        print(f"  [tracker] Warning: no se pudo guardar tracker_metrics.json: {exc}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_tracker(quiet: bool = False, no_update: bool = False, no_report: bool = False) -> None:
    """
    Core tracker logic. Can be called programmatically from main.py.
    """
    # 1. Resolve pending results
    if not no_update:
        update_results(db_path=PICKS_DB, quiet=quiet)

    # 2. Load all picks and compute metrics
    all_picks  = db_picks.get_all_picks(PICKS_DB)
    resolved   = [p for p in all_picks if p["actual_result"] is not None]
    pending    = [p for p in all_picks if p["actual_result"] is None]

    metrics = None
    if resolved:
        metrics = compute_metrics(resolved)
        metrics["n_pending"] = len(pending)
        metrics["n_total"]   = len(all_picks)

        # PSI drift detection
        psi = compute_psi(resolved)
        metrics["psi"] = psi
        if psi.get("drift_detected"):
            _alert_psi_drift(psi)
        elif not quiet and psi.get("psi_max") is not None:
            print(f"  [tracker] PSI={psi['psi_max']:.3f} (drift: no)")

        # Bet timing analysis
        metrics["timing_analysis"] = compute_timing_analysis(resolved)

        # Sub-model accuracy (rolling window)
        metrics["submodel_accuracy_30"] = compute_submodel_accuracy(resolved)

    # 3. Maybe update calibrator, optimise weights, train meta-learner, draw model
    if resolved:
        maybe_fit_calibrator(resolved)
        maybe_optimize_weights(resolved)
        maybe_train_meta_learner(resolved)
        try:
            from algorithms.draw_model import train as _train_draw
            _draw_result = _train_draw(PICKS_DB)
            if not quiet:
                print(f"  [tracker] {_draw_result}")
        except Exception as _e:
            if not quiet:
                print(f"  [tracker] draw_model: {_e}")

        try:
            from algorithms.over25_model import train as _train_o25
            _o25_result = _train_o25(PICKS_DB)
            if not quiet:
                print(f"  [tracker] {_o25_result}")
        except Exception as _e:
            if not quiet:
                print(f"  [tracker] over25_model: {_e}")

    # 4. Save metrics snapshot for downstream consumers (e.g. dynamic Kelly)
    if metrics:
        _save_metrics_json(metrics)

    # 5. Generate tracker_data.js + results.js (for frontend bet auto-settlement)
    if not no_report:
        path = generate_tracker_js(all_picks, metrics)
        _save_results_js(all_picks)
        if not quiet:
            print(f"  [tracker] {path} generado "
                  f"({len(resolved)} resueltos · {len(pending)} pendientes).")


def main():
    parser = argparse.ArgumentParser(description="BetWinninGames — Tracker P&L")
    parser.add_argument("--no-update", action="store_true",
                        help="Skip API calls; regenerate JS from existing DB only")
    parser.add_argument("--no-report", action="store_true",
                        help="Update results but skip JS generation")
    args = parser.parse_args()

    run_tracker(quiet=False, no_update=args.no_update, no_report=args.no_report)


if __name__ == "__main__":
    main()
