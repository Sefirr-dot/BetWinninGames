"""
Ensemble weight optimizer.

Uses resolved picks from picks_history.db to find the DC/Elo/Form weight
combination that minimises the Brier score on historical data.

Only the three main 1X2 models are optimised (Dixon-Coles, Elo, Form).
H2H / BTTS / Corners weights remain at their config values.

The optimised weights are stored at cache/model_weights.json and loaded by
ensemble.py at import time, overriding config.MODEL_WEIGHTS for the three
main models.
"""

import json
import os

import numpy as np
from scipy.optimize import minimize

from config import MODEL_WEIGHTS, WEIGHT_OPTIMIZER_MIN_SAMPLES

_WEIGHTS_PATH = "cache/model_weights.json"


# ---------------------------------------------------------------------------
# Core optimisation
# ---------------------------------------------------------------------------

def optimize_weights(resolved_picks: list[dict]) -> dict | None:
    """
    Find DC/Elo/Form weights that minimise Brier score on resolved picks.

    Parameters
    ----------
    resolved_picks : list of DB row dicts with sub_preds (JSON) and actual_result

    Returns
    -------
    Full MODEL_WEIGHTS-compatible dict with optimised DC/Elo/Form values,
    or None if not enough usable records.
    """
    records = []
    for p in resolved_picks:
        raw = p.get("sub_preds")
        if not raw:
            continue
        try:
            sub = json.loads(raw)
        except (TypeError, ValueError):
            continue

        dc   = sub.get("dc")
        elo  = sub.get("elo")
        form = sub.get("form")
        if not (dc and elo and form):
            continue

        outcome = p.get("actual_result")
        if outcome not in ("home", "draw", "away"):
            continue

        records.append({
            "dc":   [dc.get("prob_home",   1/3), dc.get("prob_draw",   1/3), dc.get("prob_away",   1/3)],
            "elo":  [elo.get("prob_home",  1/3), elo.get("prob_draw",  1/3), elo.get("prob_away",  1/3)],
            "form": [form.get("prob_home", 1/3), form.get("prob_draw", 1/3), form.get("prob_away", 1/3)],
            "y":    [
                1.0 if outcome == "home" else 0.0,
                1.0 if outcome == "draw" else 0.0,
                1.0 if outcome == "away" else 0.0,
            ],
        })

    if len(records) < WEIGHT_OPTIMIZER_MIN_SAMPLES:
        return None

    dc_arr   = np.array([r["dc"]   for r in records])
    elo_arr  = np.array([r["elo"]  for r in records])
    form_arr = np.array([r["form"] for r in records])
    y_arr    = np.array([r["y"]    for r in records])

    def brier(w):
        w = np.abs(w)  # force non-negative
        w = w / w.sum()
        blend = w[0] * dc_arr + w[1] * elo_arr + w[2] * form_arr
        # Row-normalise
        row_sums = blend.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums < 1e-9, 1.0, row_sums)
        blend /= row_sums
        return float(np.mean(np.sum((blend - y_arr) ** 2, axis=1)))

    x0 = [
        MODEL_WEIGHTS["dixon_coles"],
        MODEL_WEIGHTS["elo"],
        MODEL_WEIGHTS["form"],
    ]
    result = minimize(
        brier,
        x0=x0,
        method="L-BFGS-B",
        bounds=[(0.05, 0.85), (0.05, 0.85), (0.05, 0.85)],
        options={"maxiter": 300, "ftol": 1e-10},
    )

    w_opt = np.abs(result.x)
    w_opt /= w_opt.sum()

    # Scale DC/Elo/Form to occupy (1 - other_weights) of total weight budget
    other = MODEL_WEIGHTS["h2h"] + MODEL_WEIGHTS["btts"] + MODEL_WEIGHTS["corners"]
    scale = 1.0 - other

    return {
        "dixon_coles": round(float(w_opt[0]) * scale, 4),
        "elo":         round(float(w_opt[1]) * scale, 4),
        "form":        round(float(w_opt[2]) * scale, 4),
        # keep the rest from config
        "h2h":     MODEL_WEIGHTS["h2h"],
        "btts":    MODEL_WEIGHTS["btts"],
        "corners": MODEL_WEIGHTS["corners"],
        "_optimised_from_n": len(records),
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
        # Validate: must have the three main keys
        if all(k in data for k in ("dixon_coles", "elo", "form")):
            return data
        return None
    except Exception:
        return None
