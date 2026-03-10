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
import sys
from datetime import date, datetime

import db_picks
import fetcher
from config import CALIBRATION_MIN_SAMPLES, PICKS_DB, TRACKER_JS_PATH


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

_METRICS_JSON_PATH = "cache/tracker_metrics.json"

def _save_metrics_json(metrics: dict) -> None:
    """
    Persist a lightweight snapshot of tracker metrics to cache/tracker_metrics.json.
    Used by value_detector.py to load per-league ROI for dynamic Kelly sizing.
    Only saves the fields that downstream consumers need (avoids serialising
    the full bankroll_history list which can be large).
    """
    snapshot = {
        "n_resolved":    metrics.get("n_resolved", 0),
        "accuracy_1x2":  metrics.get("accuracy_1x2"),
        "roi_flat":      metrics.get("roi_flat"),
        "per_league":    metrics.get("per_league", {}),
        "per_stars":     metrics.get("per_stars", {}),
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

    # 5. Generate tracker_data.js
    if not no_report:
        path = generate_tracker_js(all_picks, metrics)
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
