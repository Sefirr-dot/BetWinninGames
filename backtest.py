"""
Walk-forward backtesting for BetWinninGames.

Evaluates accuracy, Brier Score, log-loss and flat ROI on historical data
without data leakage: models are re-fitted on matches up to each test fold.

Usage:
    python backtest.py --league PL --seasons 2023 2024
    python backtest.py --league PL --seasons 2024 --min-train 150
"""

import argparse
import json
import math
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
import sys
from datetime import date, datetime

import db_picks
import fetcher
import fdco_fetcher
import understat_fetcher
from algorithms import dixon_coles, elo as elo_module, ensemble
from config import LEAGUES, BACKTEST_MIN_TRAIN, BACKTEST_BATCH_SIZE, BACKTEST_JS_PATH, PICKS_DB


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_matches(league: str, seasons: list[int]) -> list[dict]:
    """Load and sort all FINISHED matches for the given league + seasons."""
    comp_id = LEAGUES.get(league)
    if not comp_id:
        raise ValueError(f"Unknown league '{league}'. Valid: {list(LEAGUES.keys())}")

    all_matches = []
    for season in seasons:
        print(f"  [backtest] Loading {league} season {season}...")
        matches = fetcher.get_season_matches(comp_id, season)
        for m in matches:
            m["_league_code"] = league
        all_matches.extend(matches)
        print(f"             {len(matches)} matches loaded.")

    all_matches.sort(key=lambda m: m.get("utcDate", ""))
    return all_matches


def _actual_result(match: dict) -> str:
    ft = match.get("score", {}).get("fullTime", {})
    home = ft.get("home")
    away = ft.get("away")
    if home is None or away is None:
        return "unknown"
    if home > away:
        return "home"
    elif home < away:
        return "away"
    return "draw"


def _actual_over25(match: dict) -> bool:
    ft = match.get("score", {}).get("fullTime", {})
    h = ft.get("home") or 0
    a = ft.get("away") or 0
    return (h + a) > 2


def _actual_btts(match: dict) -> bool:
    ft = match.get("score", {}).get("fullTime", {})
    h = ft.get("home") or 0
    a = ft.get("away") or 0
    return h > 0 and a > 0


# ---------------------------------------------------------------------------
# Walk-forward engine
# ---------------------------------------------------------------------------

def run_backtest(
    matches: list[dict],
    min_train: int = BACKTEST_MIN_TRAIN,
    batch_size: int = BACKTEST_BATCH_SIZE,
) -> list[dict]:
    """
    Walk-forward backtest.

    Folds:
        train = matches[:train_end]
        test  = matches[train_end : train_end + batch_size]

    Models are re-fitted per fold; test matches never appear in training data.

    Returns a list of result dicts, one per successfully predicted match.
    """
    results = []
    train_end = min_train
    fold = 0

    while train_end < len(matches):
        train_set = matches[:train_end]
        test_set  = matches[train_end : train_end + batch_size]

        if not test_set:
            break

        fold += 1
        test_start_date = test_set[0].get("utcDate", "?")[:10]
        print(f"  Fold {fold}: train={len(train_set)}, test={len(test_set)}, "
              f"test_from={test_start_date}")

        # Reference date = day of first test match (no future leakage)
        ref_date = datetime.strptime(test_start_date, "%Y-%m-%d").date()

        # Fit models on training data only
        dc_params      = dixon_coles.fit_per_league(train_set, reference_date=ref_date)
        elo_ratings    = elo_module.build_ratings(train_set)
        elo_home_rt, elo_away_rt = elo_module.build_split_ratings(train_set)

        for match in test_set:
            home_id = match.get("homeTeam", {}).get("id")
            away_id = match.get("awayTeam", {}).get("id")
            if home_id is None or away_id is None:
                continue

            actual = _actual_result(match)
            if actual == "unknown":
                continue

            try:
                match_date  = datetime.strptime(match["utcDate"][:10], "%Y-%m-%d").date()
                league_code = match.get("_league_code")

                # Bookmaker odds from fdco CSV (Pinnacle preferred, B365 fallback)
                market_odds = None
                bk_h = match.get("_bk_h")
                bk_d = match.get("_bk_d")
                bk_a = match.get("_bk_a")
                if bk_h and bk_d and bk_a and bk_h > 1.0 and bk_d > 1.0 and bk_a > 1.0:
                    market_odds = {"odds_1": bk_h, "odds_x": bk_d, "odds_2": bk_a}

                pred = ensemble.predict_match(
                    home_id, away_id, dc_params, elo_ratings, train_set,
                    reference_date=match_date,
                    league_code=league_code,
                    elo_home_ratings=elo_home_rt,
                    elo_away_ratings=elo_away_rt,
                    market_odds=market_odds,
                )
                tc = match.get("_total_corners")
                results.append({
                    "fold_id":        fold,
                    "match":          match,
                    "prediction":     pred,
                    "actual":         actual,
                    "actual_over25":  _actual_over25(match),
                    "actual_btts":    _actual_btts(match),
                    "actual_corners": tc,   # None when not available (non-fdco matches)
                    "market_odds":    market_odds,
                })
            except Exception as e:
                home = match.get("homeTeam", {}).get("name", "?")
                away = match.get("awayTeam", {}).get("name", "?")
                print(f"    [WARNING] {home} vs {away}: {e}")

        train_end += batch_size

    return results


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(results: list[dict]) -> dict:
    """
    Compute accuracy, Brier Score, log-loss, flat ROI and calibration.

    Returns
    -------
    dict with keys: n_matches, accuracy_1x2, brier_score, log_loss,
                    roi_flat, accuracy_over25, accuracy_btts, calibration
    """
    if not results:
        return {}

    n = len(results)

    # 1X2 accuracy
    correct_1x2 = sum(1 for r in results if r["prediction"]["best_outcome"] == r["actual"])

    # Brier score (multi-class) and log-loss
    brier = 0.0
    log_loss_val = 0.0
    for r in results:
        pred   = r["prediction"]
        actual = r["actual"]
        probs  = {
            "home": pred["prob_home"],
            "draw": pred["prob_draw"],
            "away": pred["prob_away"],
        }
        for outcome, p in probs.items():
            actual_bin = 1.0 if actual == outcome else 0.0
            brier += (p - actual_bin) ** 2
        log_loss_val -= math.log(max(probs.get(actual, 1e-9), 1e-9))

    brier /= n
    log_loss_val /= n

    # Flat ROI: £1 on best_outcome at fair odds (1 / model_prob)
    roi_flat = 0.0
    for r in results:
        pred     = r["prediction"]
        fair_odds = 1.0 / max(pred["best_prob"], 1e-9)
        if pred["best_outcome"] == r["actual"]:
            roi_flat += fair_odds - 1.0   # net win
        else:
            roi_flat -= 1.0               # stake lost
    roi_flat /= n

    # Value bet ROI: only matches where bookmaker odds existed + model edge > 0
    vb_results = []
    for r in results:
        mkt = r.get("market_odds")
        if not mkt:
            continue
        pred = r["prediction"]
        outcome = pred["best_outcome"]
        odds_map = {"home": mkt["odds_1"], "draw": mkt["odds_x"], "away": mkt["odds_2"]}
        bk_odds = odds_map.get(outcome, 0.0)
        if bk_odds <= 1.0:
            continue
        implied = 1.0 / bk_odds
        model_p = pred["best_prob"]
        edge    = model_p - implied
        if edge > 0:
            vb_results.append(r)
    vb_n = len(vb_results)
    vb_roi = None
    vb_acc = None
    if vb_n:
        vb_correct = sum(1 for r in vb_results if r["prediction"]["best_outcome"] == r["actual"])
        vb_acc = vb_correct / vb_n
        vb_profit = 0.0
        for r in vb_results:
            mkt = r["market_odds"]
            outcome = r["prediction"]["best_outcome"]
            bk_odds = {"home": mkt["odds_1"], "draw": mkt["odds_x"], "away": mkt["odds_2"]}[outcome]
            if r["prediction"]["best_outcome"] == r["actual"]:
                vb_profit += bk_odds - 1.0
            else:
                vb_profit -= 1.0
        vb_roi = vb_profit / vb_n

    # Over 2.5 and BTTS accuracy
    correct_over25 = sum(
        1 for r in results if (r["prediction"]["over25"] >= 0.5) == r["actual_over25"]
    )
    correct_btts = sum(
        1 for r in results if (r["prediction"]["btts_prob"] >= 0.5) == r["actual_btts"]
    )

    # Calibration by best_prob bucket (1X2)
    buckets = [(0.3, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 1.01)]
    calibration = {}
    for lo, hi in buckets:
        label = f"{lo:.1f}-{hi:.1f}" if hi < 1.0 else f"{lo:.1f}+"
        bucket = [r for r in results if lo <= r["prediction"]["best_prob"] < hi]
        if bucket:
            win_rate = sum(
                1 for r in bucket if r["prediction"]["best_outcome"] == r["actual"]
            ) / len(bucket)
            avg_prob = sum(r["prediction"]["best_prob"] for r in bucket) / len(bucket)
            calibration[label] = {
                "n":           len(bucket),
                "avg_prob":    avg_prob,
                "actual_rate": win_rate,
            }

    # Calibration by Over 2.5 probability bucket
    calibration_over25 = {}
    for lo, hi in buckets:
        label = f"{lo:.1f}-{hi:.1f}" if hi < 1.0 else f"{lo:.1f}+"
        bucket = [r for r in results if lo <= r["prediction"]["over25"] < hi]
        if bucket:
            actual_rate = sum(1 for r in bucket if r["actual_over25"]) / len(bucket)
            avg_prob    = sum(r["prediction"]["over25"] for r in bucket) / len(bucket)
            calibration_over25[label] = {
                "n":           len(bucket),
                "avg_prob":    avg_prob,
                "actual_rate": actual_rate,
            }

    # Calibration by BTTS probability bucket
    calibration_btts = {}
    for lo, hi in buckets:
        label = f"{lo:.1f}-{hi:.1f}" if hi < 1.0 else f"{lo:.1f}+"
        bucket = [r for r in results if lo <= r["prediction"]["btts_prob"] < hi]
        if bucket:
            actual_rate = sum(1 for r in bucket if r["actual_btts"]) / len(bucket)
            avg_prob    = sum(r["prediction"]["btts_prob"] for r in bucket) / len(bucket)
            calibration_btts[label] = {
                "n":           len(bucket),
                "avg_prob":    avg_prob,
                "actual_rate": actual_rate,
            }

    # Per-league Over 2.5 and BTTS accuracy
    from collections import defaultdict
    _lg_o25:  dict[str, list] = defaultdict(list)
    _lg_btts: dict[str, list] = defaultdict(list)
    for r in results:
        lg = r["match"].get("_league_code", "")
        if not lg:
            continue
        _lg_o25[lg].append((r["prediction"]["over25"] >= 0.5, r["actual_over25"]))
        _lg_btts[lg].append((r["prediction"]["btts_prob"] >= 0.5, r["actual_btts"]))

    per_league_over25 = {
        lg: {
            "n":        len(items),
            "accuracy": round(sum(1 for p, a in items if p == a) / len(items), 4),
        }
        for lg, items in _lg_o25.items() if items
    }
    per_league_btts = {
        lg: {
            "n":        len(items),
            "accuracy": round(sum(1 for p, a in items if p == a) / len(items), 4),
        }
        for lg, items in _lg_btts.items() if items
    }

    # Corners validation (only for matches where actual corners are available)
    corners_results = [
        r for r in results
        if r.get("actual_corners") is not None
        and r["prediction"].get("corners", {}).get("expected_corners") is not None
    ]
    corners_mae      = None
    corners_accuracy = None
    corners_n        = len(corners_results)
    if corners_n:
        mae_sum = sum(
            abs(r["prediction"]["corners"]["expected_corners"] - r["actual_corners"])
            for r in corners_results
        )
        corners_mae = mae_sum / corners_n

        # Accuracy on the 9.5 over/under line (same line used by reporter/visualizer)
        correct_corners = sum(
            1 for r in corners_results
            if (r["prediction"]["corners"].get("over_lines", {}).get(9.5, 0) >= 0.5)
               == (r["actual_corners"] > 9.5)
        )
        corners_accuracy = correct_corners / corners_n

    return {
        "n_matches":          n,
        "accuracy_1x2":       correct_1x2 / n,
        "brier_score":        brier,
        "log_loss":           log_loss_val,
        "roi_flat":           roi_flat,
        "accuracy_over25":    correct_over25 / n,
        "accuracy_btts":      correct_btts / n,
        "calibration":        calibration,
        "calibration_over25": calibration_over25,
        "calibration_btts":   calibration_btts,
        "per_league_over25":  per_league_over25,
        "per_league_btts":    per_league_btts,
        "corners_mae":        round(corners_mae, 3)      if corners_mae      is not None else None,
        "corners_accuracy":   round(corners_accuracy, 4) if corners_accuracy is not None else None,
        "corners_n":          corners_n,
        "vb_n":               vb_n,
        "vb_accuracy":        vb_acc,
        "vb_roi":             vb_roi,
    }


# ---------------------------------------------------------------------------
# Per-fold metrics
# ---------------------------------------------------------------------------

def compute_fold_metrics(results: list[dict]) -> list[dict]:
    """
    Group results by fold_id and compute lightweight metrics per fold.

    Returns a list of dicts sorted by fold_id:
      fold, test_from, n_train (from first result in fold), n_test,
      accuracy_1x2, roi_flat, brier_score
    """
    from collections import defaultdict
    buckets: dict[int, list[dict]] = defaultdict(list)
    for r in results:
        buckets[r.get("fold_id", 0)].append(r)

    folds = []
    for fold_id in sorted(buckets):
        fold_results = buckets[fold_id]
        n = len(fold_results)
        if n == 0:
            continue

        test_from = fold_results[0]["match"].get("utcDate", "")[:10]

        correct = sum(1 for r in fold_results
                      if r["prediction"]["best_outcome"] == r["actual"])
        accuracy = correct / n

        roi = 0.0
        for r in fold_results:
            pred = r["prediction"]
            fair_odds = 1.0 / max(pred["best_prob"], 1e-9)
            roi += (fair_odds - 1.0) if pred["best_outcome"] == r["actual"] else -1.0
        roi /= n

        brier = 0.0
        for r in fold_results:
            pred = r["prediction"]
            probs = {"home": pred["prob_home"], "draw": pred["prob_draw"], "away": pred["prob_away"]}
            for outcome, p in probs.items():
                brier += (p - (1.0 if r["actual"] == outcome else 0.0)) ** 2
        brier /= n

        folds.append({
            "fold":        fold_id,
            "test_from":   test_from,
            "n_test":      n,
            "accuracy":    round(accuracy, 4),
            "roi_flat":    round(roi, 4),
            "brier_score": round(brier, 4),
        })

    return folds


# ---------------------------------------------------------------------------
# Report generator
# ---------------------------------------------------------------------------

def generate_report(
    metrics: dict,
    results: list[dict],
    league: str,
    seasons: list[int],
    fold_stats: list[dict] | None = None,
) -> str:
    """Write backtest_YYYY-MM-DD.txt and return the file path."""
    output_path = f"backtest_{datetime.now().strftime('%Y-%m-%d')}.txt"

    lines = []

    def w(s=""):
        lines.append(s)

    sep = "═" * 60
    thin = "─" * 40

    w(sep)
    w("  BETWINNINGAMES — BACKTESTING WALK-FORWARD")
    w(f"  Liga: {league}  |  Temporadas: {', '.join(str(s) for s in seasons)}")
    w(f"  Partidos evaluados: {metrics['n_matches']}")
    w(sep)
    w()

    w("MÉTRICAS GENERALES")
    w(thin)
    w(f"  Accuracy 1X2:      {metrics['accuracy_1x2'] * 100:.1f}%")
    w(f"  Brier Score:       {metrics['brier_score']:.4f}  (↓ mejor, random ≈ 0.667)")
    w(f"  Log Loss:          {metrics['log_loss']:.4f}  (↓ mejor)")
    w(f"  ROI Flat:          {metrics['roi_flat'] * 100:+.1f}%  (a cuotas justas del modelo)")
    w(f"  Accuracy Over 2.5: {metrics['accuracy_over25'] * 100:.1f}%")
    w(f"  Accuracy BTTS:     {metrics['accuracy_btts'] * 100:.1f}%")
    if metrics.get("corners_n"):
        mae = metrics["corners_mae"]
        acc = metrics["corners_accuracy"]
        verdict = "OK (<1.8)" if mae < 1.8 else "MARGINAL (1.8-2.5)" if mae < 2.5 else "REDISENAR (>2.5)"
        w(f"  Corners MAE:       {mae:.3f} corners  [{verdict}]  (n={metrics['corners_n']})")
        w(f"  Corners Acc O9.5:  {acc * 100:.1f}%")
    if metrics.get("vb_n"):
        vb_roi_str = f"{metrics['vb_roi'] * 100:+.1f}%" if metrics["vb_roi"] is not None else "N/A"
        vb_acc_str = f"{metrics['vb_accuracy'] * 100:.1f}%" if metrics["vb_accuracy"] is not None else "N/A"
        w(f"  Value Bets (n={metrics['vb_n']}):  ROI={vb_roi_str}  Acc={vb_acc_str}  (cuotas bookmaker reales)")
    w()

    # Per-fold breakdown
    if fold_stats:
        w("RENDIMIENTO POR FOLD  (degradacion temporal del modelo)")
        w(thin)
        w(f"  {'Fold':>4}  {'Desde':>10}  {'N':>4}  {'Accuracy':>9}  {'ROI Flat':>9}  {'Brier':>7}")
        w("  " + "─" * 52)
        for fs in fold_stats:
            w(
                f"  {fs['fold']:>4}  {fs['test_from']:>10}  {fs['n_test']:>4}"
                f"  {fs['accuracy'] * 100:>8.1f}%"
                f"  {fs['roi_flat'] * 100:>+8.1f}%"
                f"  {fs['brier_score']:>7.4f}"
            )
        w()

    def _write_calibration_table(title: str, cal: dict) -> None:
        if not cal:
            return
        w(title)
        w(thin)
        w(f"  {'Bucket':<12} {'N':>6} {'Prob media':>12} {'Tasa real':>12} {'Diferencia':>12}")
        w("  " + "─" * 56)
        for bucket, data in sorted(cal.items()):
            diff = data["actual_rate"] - data["avg_prob"]
            w(
                f"  {bucket:<12} {data['n']:>6} "
                f"{data['avg_prob'] * 100:>11.1f}% "
                f"{data['actual_rate'] * 100:>11.1f}% "
                f"{diff * 100:>+11.1f}%"
            )
        w()

    _write_calibration_table("CALIBRACIÓN 1X2  (prob predicha vs tasa real de acierto)", metrics["calibration"])
    _write_calibration_table("CALIBRACIÓN OVER 2.5", metrics.get("calibration_over25", {}))
    _write_calibration_table("CALIBRACIÓN BTTS", metrics.get("calibration_btts", {}))

    if metrics.get("per_league_over25") or metrics.get("per_league_btts"):
        w("ACCURACY POR LIGA — MERCADOS SECUNDARIOS")
        w(thin)
        w(f"  {'Liga':<6} {'Over2.5 Acc':>12} {'Over2.5 N':>10} {'BTTS Acc':>10} {'BTTS N':>8}")
        w("  " + "─" * 52)
        all_leagues = sorted(set(
            list(metrics.get("per_league_over25", {}).keys()) +
            list(metrics.get("per_league_btts", {}).keys())
        ))
        for lg in all_leagues:
            o25_d  = metrics.get("per_league_over25", {}).get(lg, {})
            btts_d = metrics.get("per_league_btts",   {}).get(lg, {})
            o25_acc  = f"{o25_d['accuracy']*100:.1f}%"  if o25_d  else "—"
            btts_acc = f"{btts_d['accuracy']*100:.1f}%" if btts_d else "—"
            w(f"  {lg:<6} {o25_acc:>12} {o25_d.get('n','—'):>10} {btts_acc:>10} {btts_d.get('n','—'):>8}")
        w()

    w("MUESTRA — PRIMEROS 15 RESULTADOS")
    w(thin)
    w(f"  {'Partido':<30} {'Pred':>5} {'Real':>5} {'Prob':>7}  OK")
    w("  " + "─" * 55)
    for r in results[:15]:
        match = r["match"]
        home = (match.get("homeTeam", {}).get("shortName")
                or match.get("homeTeam", {}).get("name", "?"))[:12]
        away = (match.get("awayTeam", {}).get("shortName")
                or match.get("awayTeam", {}).get("name", "?"))[:12]
        match_str   = f"{home} vs {away}"
        pred_label  = r["prediction"]["best_outcome"][:4].upper()
        actual_label = r["actual"][:4].upper()
        prob        = r["prediction"]["best_prob"] * 100
        ok          = "✓" if pred_label == actual_label else "✗"
        w(f"  {match_str:<30} {pred_label:>5} {actual_label:>5} {prob:>6.1f}%  {ok}")
    w()

    w(sep)
    w(f"  Generado: {datetime.now().strftime('%Y-%m-%d %H:%M')} | Datos: football-data.org")
    w(sep)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return output_path


# ---------------------------------------------------------------------------
# DB seeder
# ---------------------------------------------------------------------------

def seed_picks_db(results: list[dict], db_path: str = PICKS_DB) -> int:
    """
    Insert walk-forward backtest results into picks_history.db as resolved picks.

    Each result is saved via INSERT OR IGNORE (safe to re-run), then immediately
    marked as resolved with the actual match outcome.  Picks that already exist
    in the DB (e.g. from real main.py runs) are left untouched.

    Returns the number of new rows inserted.
    """
    run_ts = datetime.now().isoformat()
    total = 0

    for r in results:
        match = r["match"]
        match_id = match.get("id")
        if match_id is None:
            continue

        match_date = match.get("utcDate", "")[:10]
        wrapped = {"match_info": match, "prediction": r["prediction"]}

        n = db_picks.save_picks([wrapped], match_date, run_ts, db_path, source="backtest")
        total += n

        # Only write the result for rows we just inserted — never overwrite
        # a real result already in the DB from tracker.py
        if n > 0:
            db_picks.update_result(
                match_id,
                r["actual"],
                int(r["actual_over25"]),
                int(r["actual_btts"]),
                db_path,
            )

    return total


# ---------------------------------------------------------------------------
# JS export
# ---------------------------------------------------------------------------

def generate_backtest_js(
    metrics: dict,
    results: list[dict],
    league: str,
    seasons: list[int],
    output_path: str = BACKTEST_JS_PATH,
    fold_stats: list[dict] | None = None,
    per_league: dict | None = None,
) -> str:
    """Write visualizador/data/backtest_data.js and return the path."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Walk-forward bankroll + accuracy curve (sorted by match date)
    results_sorted = sorted(results, key=lambda r: r["match"].get("utcDate", ""))
    n_total    = max(len(results_sorted), 1)
    unit_stake = 1.0 / n_total   # proportional stake: curve stays near 1.0
    bankroll   = 1.0
    correct    = 0
    curve      = []
    for i, r in enumerate(results_sorted):
        pred   = r["prediction"]
        actual = r["actual"]
        fair   = 1.0 / max(pred["best_prob"], 1e-9)
        if pred["best_outcome"] == actual:
            bankroll += unit_stake * (fair - 1.0)
            correct  += 1
        else:
            bankroll -= unit_stake
        curve.append({
            "date":     r["match"].get("utcDate", "")[:10],
            "bankroll": round(bankroll, 4),
            "acc":      round(correct / (i + 1) * 100, 1),
            "league":   r["match"].get("_league_code", ""),
        })

    data = {
        "league":      league,
        "seasons":     seasons,
        "generatedAt": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "metrics":     metrics,
        "curve":       curve,
        "folds":       fold_stats or [],
        "per_league":  per_league or {},
    }

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(
            "// Auto-generated by BetWinninGames backtest\n"
            f"var BACKTEST_DATA = {json.dumps(data, ensure_ascii=False, indent=2)};\n"
        )

    return output_path


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="BetWinninGames Walk-Forward Backtester"
    )
    parser.add_argument(
        "--league",
        required=True,
        choices=list(LEAGUES.keys()) + ["ALL"],
        help="League to backtest (PL, PD, BL1, FL1) or ALL to run every league",
    )
    parser.add_argument(
        "--seasons",
        nargs="+",
        type=int,
        required=True,
        help="One or more seasons (e.g. 2023 2024)",
    )
    parser.add_argument(
        "--min-train",
        type=int,
        default=BACKTEST_MIN_TRAIN,
        help=f"Minimum training matches before first test fold (default {BACKTEST_MIN_TRAIN})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BACKTEST_BATCH_SIZE,
        help=f"Matches per walk-forward fold (default {BACKTEST_BATCH_SIZE})",
    )
    parser.add_argument(
        "--seed-db",
        action="store_true",
        help="Seed picks_history.db with backtest results (marked as already resolved). "
             "Safe to re-run: uses INSERT OR IGNORE.",
    )
    return parser.parse_args()


def _run_league(league: str, seasons: list[int], min_train: int, batch_size: int) -> list[dict]:
    """Load, augment, enrich and run walk-forward for a single league. Returns results list."""
    matches = load_matches(league, seasons)
    if len(matches) < min_train:
        print(f"  [SKIP] {league}: solo {len(matches)} partidos (necesita {min_train}).")
        return []
    print(f"  {len(matches)} partidos cargados — augmenting fdco...")
    matches = fdco_fetcher.augment_historical(matches, league)
    print(f"  {len(matches)} total tras augment — enriqueciendo con xG...")
    understat_fetcher.enrich_with_xg(matches)
    print(f"  Walk-forward {league}...\n")
    return run_backtest(matches, min_train, batch_size)


def _print_summary(metrics: dict, label: str) -> None:
    print(f"  [{label}] Accuracy={metrics['accuracy_1x2']*100:.1f}%  "
          f"Brier={metrics['brier_score']:.4f}  ROI={metrics['roi_flat']*100:+.1f}%", end="")
    if metrics.get("vb_n"):
        vb = metrics["vb_roi"]
        print(f"  VB-ROI={vb*100:+.1f}% (n={metrics['vb_n']})", end="")
    print()


def main():
    args = parse_args()

    leagues_to_run = list(LEAGUES.keys()) if args.league == "ALL" else [args.league]

    print(f"\n{'='*60}")
    print(f"  BetWinninGames Backtest — {args.league} {args.seasons}")
    print(f"  min_train={args.min_train}  batch_size={args.batch_size}")
    print(f"{'='*60}\n")

    all_results: list[dict] = []
    per_league_metrics: dict = {}

    for league in leagues_to_run:
        print(f"\n--- {league} ---")
        results = _run_league(league, args.seasons, args.min_train, args.batch_size)
        if not results:
            continue
        lm = compute_metrics(results)
        per_league_metrics[league] = {
            "n_matches":    lm["n_matches"],
            "accuracy_1x2": round(lm["accuracy_1x2"], 4),
            "brier_score":  round(lm["brier_score"],  4),
            "roi_flat":     round(lm["roi_flat"],      4),
            "vb_n":         lm.get("vb_n"),
            "vb_roi":       lm.get("vb_roi"),
        }
        _print_summary(lm, league)
        all_results.extend(results)

    if not all_results:
        print("[WARNING] Sin resultados. Verifica disponibilidad de datos.")
        sys.exit(0)

    print(f"\n  {len(all_results)} predicciones totales — calculando métricas globales...")
    metrics    = compute_metrics(all_results)
    fold_stats = compute_fold_metrics(all_results)

    league_label = "ALL" if args.league == "ALL" else args.league
    output_path  = generate_report(metrics, all_results, league_label, args.seasons, fold_stats)
    js_path      = generate_backtest_js(
        metrics, all_results, league_label, args.seasons,
        fold_stats=fold_stats,
        per_league=per_league_metrics,
    )

    print(f"\n{'='*60}")
    if args.league == "ALL":
        print("  RESUMEN POR LIGA:")
        for lg, lm in per_league_metrics.items():
            _print_summary(lm, lg)
        print()
    print(f"  GLOBAL — Accuracy={metrics['accuracy_1x2']*100:.1f}%  "
          f"Brier={metrics['brier_score']:.4f}  ROI={metrics['roi_flat']*100:+.1f}%")
    if metrics.get("vb_n"):
        vb_roi_str = f"{metrics['vb_roi'] * 100:+.1f}%" if metrics["vb_roi"] is not None else "N/A"
        print(f"  VB ROI (bk):   {vb_roi_str}  (n={metrics['vb_n']} value bets)")
    print(f"  Folds:         {len(fold_stats)}")
    print(f"  Informe:       {output_path}")
    print(f"  JS:            {js_path}")

    # Always pretrain draw model from backtest results (cheap, no DB required).
    # Skips automatically if a live-trained model already exists.
    try:
        from algorithms.draw_model import pretrain_from_backtest as _pretrain_draw
        _dr = _pretrain_draw(all_results)
        print(f"  Draw model:    {_dr}")
    except Exception as _e:
        print(f"  Draw model:    error — {_e}")

    if args.seed_db:
        print(f"\n  Sembrando picks_history.db...")
        n_seeded = seed_picks_db(all_results)
        print(f"  {n_seeded} picks nuevos insertados en {PICKS_DB}")

        from algorithms import meta_learner
        ml_result = meta_learner.train(PICKS_DB)
        if "error" in ml_result:
            print(f"  Meta-learner: {ml_result['error']}")
        else:
            print(f"  Meta-learner XGBoost entrenado  "
                  f"n={ml_result['n_samples']}  "
                  f"best_round={ml_result.get('best_round','?')}  "
                  f"Brier_train={ml_result['brier_train']:.4f}  "
                  f"Brier_val={ml_result.get('brier_val','?')}")

    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
