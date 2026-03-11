"""
Ensemble weight optimizer.

Uses resolved picks from picks_history.db to find the DC/Elo weight
combination that maximises average CLV vs Pinnacle closing line (primary
objective) or minimises Brier score (fallback when CLV data is scarce).

CLV objective (primary):
    CLV = our_implied_prob - 1/pinnacle_closing_odds
    Maximise mean(CLV) over picks with closing_odds populated.
    A positive avg CLV means we consistently predict better than the sharp
    Pinnacle closing line — the gold standard for edge validation.

Brier fallback:
    Used when fewer than CLV_OPTIMIZER_MIN_SAMPLES picks have CLV data,
    or when WEIGHT_OPTIMIZER_OBJECTIVE = "brier" in config.

Form has been removed from the optimisation — backtest confirmed near-zero
predictive value (3.4% weight), so its budget is now pre-allocated to DC/Elo.

Only DC and Elo are optimised. H2H / BTTS / Corners weights remain at their
config values.

The optimised weights are stored at cache/model_weights.json and loaded by
ensemble.py at import time, overriding config.MODEL_WEIGHTS.
"""

import json
import os

import numpy as np
from scipy.optimize import minimize

from config import (MODEL_WEIGHTS, WEIGHT_OPTIMIZER_MIN_SAMPLES,
                    WEIGHT_OPTIMIZER_OBJECTIVE, CLV_OPTIMIZER_MIN_SAMPLES)

_WEIGHTS_PATH = "cache/model_weights.json"


# ---------------------------------------------------------------------------
# Core optimisation
# ---------------------------------------------------------------------------

def optimize_weights(resolved_picks: list[dict]) -> dict | None:
    """
    Find DC/Elo weights that maximise avg CLV (primary) or minimise Brier (fallback).
    Form is fixed at 0 (removed from blend).

    The objective used is logged in the returned dict under "_objective".

    Returns
    -------
    Full MODEL_WEIGHTS-compatible dict with optimised DC/Elo values,
    or None if not enough usable records.
    """
    # ── Build per-pick records ────────────────────────────────────────────
    records = []
    clv_records = []

    for p in resolved_picks:
        raw = p.get("sub_preds")
        if not raw:
            continue
        try:
            sub = json.loads(raw)
        except (TypeError, ValueError):
            continue

        dc  = sub.get("dc")
        elo = sub.get("elo")
        if not (dc and elo):
            continue

        outcome = p.get("actual_result")
        if outcome not in ("home", "draw", "away"):
            continue

        outcome_idx = {"home": 0, "draw": 1, "away": 2}[outcome]

        record = {
            "dc":          [dc.get("prob_home",  1/3), dc.get("prob_draw",  1/3), dc.get("prob_away",  1/3)],
            "elo":         [elo.get("prob_home", 1/3), elo.get("prob_draw", 1/3), elo.get("prob_away", 1/3)],
            "y":           [
                1.0 if outcome == "home" else 0.0,
                1.0 if outcome == "draw" else 0.0,
                1.0 if outcome == "away" else 0.0,
            ],
            "outcome_idx": outcome_idx,
        }
        records.append(record)

        # CLV record: needs closing_odds (Pinnacle) populated
        closing = p.get("closing_odds")
        our_impl = p.get("our_implied_prob")
        if closing and closing > 1.0 and our_impl:
            # The closing_odds was recorded at prediction time's fair_odds;
            # CLV = our blend implied prob for best_outcome - 1/closing_odds
            record["closing_odds"] = float(closing)
            record["our_impl_prob"] = float(our_impl)
            clv_records.append(record)

    if len(records) < WEIGHT_OPTIMIZER_MIN_SAMPLES:
        return None

    dc_arr  = np.array([r["dc"]  for r in records])
    elo_arr = np.array([r["elo"] for r in records])
    y_arr   = np.array([r["y"]   for r in records])

    # ── Decide objective ───────────────────────────────────────────────────
    use_clv = (
        WEIGHT_OPTIMIZER_OBJECTIVE == "clv"
        and len(clv_records) >= CLV_OPTIMIZER_MIN_SAMPLES
    )

    if use_clv:
        clv_dc_arr  = np.array([r["dc"]  for r in clv_records])
        clv_elo_arr = np.array([r["elo"] for r in clv_records])
        clv_idx_arr = np.array([r["outcome_idx"] for r in clv_records], dtype=int)
        closing_arr = np.array([1.0 / r["closing_odds"] for r in clv_records])

        def objective(w):
            """Minimise negative avg CLV (= maximise avg CLV vs Pinnacle closing line).

            IMPORTANT: outcome_idx is recomputed dynamically under current weights
            to avoid circular bias from using historical best_outcome (which was
            computed under the old weights). This ensures the optimizer truly finds
            weights that maximise our implied prob on whatever outcome it currently
            favours, against the Pinnacle closing line.
            """
            w = np.abs(w)
            w = w / w.sum()
            blend = w[0] * clv_dc_arr + w[1] * clv_elo_arr
            row_sums = blend.sum(axis=1, keepdims=True)
            row_sums = np.where(row_sums < 1e-9, 1.0, row_sums)
            blend /= row_sums
            # Recompute best_outcome dynamically under current weights (eliminates
            # circular bias from static historical outcome labels)
            dynamic_idx = np.argmax(blend, axis=1)
            our_probs = blend[np.arange(len(clv_records)), dynamic_idx]
            clv = our_probs - closing_arr  # positive = beat sharp closing line
            return -float(np.mean(clv))   # negate: minimize = maximize CLV

        obj_label = "clv"
        n_obj = len(clv_records)
        print(f"  [weight_optimizer] Usando objetivo CLV ({n_obj} picks con cuota cierre Pinnacle).")

    else:
        if WEIGHT_OPTIMIZER_OBJECTIVE == "clv":
            print(
                f"  [weight_optimizer] Solo {len(clv_records)} picks con CLV "
                f"(min={CLV_OPTIMIZER_MIN_SAMPLES}). Usando Brier como fallback."
            )

        def objective(w):
            """Minimise Brier score (MSE on 1X2 outcomes)."""
            w = np.abs(w)
            w = w / w.sum()
            blend = w[0] * dc_arr + w[1] * elo_arr
            row_sums = blend.sum(axis=1, keepdims=True)
            row_sums = np.where(row_sums < 1e-9, 1.0, row_sums)
            blend /= row_sums
            return float(np.mean(np.sum((blend - y_arr) ** 2, axis=1)))

        obj_label = "brier"
        n_obj = len(records)

    # ── L-BFGS-B optimisation ──────────────────────────────────────────────
    x0 = [MODEL_WEIGHTS["dixon_coles"], MODEL_WEIGHTS["elo"]]
    result = minimize(
        objective,
        x0=x0,
        method="L-BFGS-B",
        bounds=[(0.10, 0.90), (0.10, 0.90)],
        options={"maxiter": 300, "ftol": 1e-10},
    )

    w_opt = np.abs(result.x)
    w_opt /= w_opt.sum()

    # Scale DC/Elo to occupy (1 - other_weights) of total weight budget
    other = MODEL_WEIGHTS["h2h"] + MODEL_WEIGHTS["btts"] + MODEL_WEIGHTS["corners"]
    scale = 1.0 - other

    return {
        "dixon_coles": round(float(w_opt[0]) * scale, 4),
        "elo":         round(float(w_opt[1]) * scale, 4),
        "form":        0.0,   # fixed at zero
        # keep the rest from config
        "h2h":     MODEL_WEIGHTS["h2h"],
        "btts":    MODEL_WEIGHTS["btts"],
        "corners": MODEL_WEIGHTS["corners"],
        "_optimised_from_n": n_obj,
        "_objective":        obj_label,
    }


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_weights(weights: dict, path: str = _WEIGHTS_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(weights, f, indent=2)


def load_weights(path: str = _WEIGHTS_PATH) -> dict | None:
    """Return optimised weights dict, or None if file doesn't exist."""
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if all(k in data for k in ("dixon_coles", "elo")):
            return data
        return None
    except Exception:
        return None
