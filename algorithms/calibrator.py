"""
Platt Scaling calibrator for ensemble 1X2 probabilities.

Fits a logistic (sigmoid) transformation per outcome:

    P_cal = 1 / (1 + exp(a * p_raw + b))

where (a, b) are found by minimising binary cross-entropy (log-loss) on the
resolved picks from picks_history.db.

The calibrator is fitted independently for home / draw / away, then the three
calibrated probabilities are renormalised to sum to 1.

Only activated when the DB contains at least CALIBRATION_MIN_SAMPLES resolved
picks (controlled by config.CALIBRATION_MIN_SAMPLES, default 200).
"""

import json
import math
import os

import numpy as np
from scipy.optimize import minimize

_DEFAULT_CALIB_PATH = "cache/calibrator.json"


# ---------------------------------------------------------------------------
# Core math
# ---------------------------------------------------------------------------

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(np.clip(x, -500, 500)))


def _log_loss(params: np.ndarray, p_raw: np.ndarray, y: np.ndarray) -> float:
    a, b = params
    p_cal = _sigmoid(a * p_raw + b)
    p_cal = np.clip(p_cal, 1e-9, 1 - 1e-9)
    return -np.mean(y * np.log(p_cal) + (1 - y) * np.log(1 - p_cal))


def _fit_one(p_raw: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Fit (a, b) for one outcome class via L-BFGS-B minimisation."""
    result = minimize(
        _log_loss,
        x0=[1.0, 0.0],
        args=(p_raw, y),
        method="L-BFGS-B",
        options={"maxiter": 500, "ftol": 1e-9},
    )
    a, b = result.x
    return float(a), float(b)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fit_calibrator(resolved_picks: list[dict]) -> dict:
    """
    Fit Platt scaling parameters from resolved picks.

    Parameters
    ----------
    resolved_picks : list of DB row dicts, each containing:
        prob_home, prob_draw, prob_away   (raw ensemble probabilities, 0-1 range)
        actual_result                     ("home" | "draw" | "away")

    Returns
    -------
    {"home": [a, b], "draw": [a, b], "away": [a, b]}
    """
    ph  = np.array([p["prob_home"]  or 0.0 for p in resolved_picks], dtype=float)
    pd_ = np.array([p["prob_draw"]  or 0.0 for p in resolved_picks], dtype=float)
    pa  = np.array([p["prob_away"]  or 0.0 for p in resolved_picks], dtype=float)

    y_h = np.array([1.0 if p["actual_result"] == "home"  else 0.0 for p in resolved_picks])
    y_d = np.array([1.0 if p["actual_result"] == "draw"  else 0.0 for p in resolved_picks])
    y_a = np.array([1.0 if p["actual_result"] == "away"  else 0.0 for p in resolved_picks])

    return {
        "home":  list(_fit_one(ph,  y_h)),
        "draw":  list(_fit_one(pd_, y_d)),
        "away":  list(_fit_one(pa,  y_a)),
    }


def apply_calibration(
    prob_home: float,
    prob_draw: float,
    prob_away: float,
    params: dict,
) -> tuple[float, float, float]:
    """
    Apply calibration parameters and renormalise to sum to 1.

    Parameters
    ----------
    prob_home, prob_draw, prob_away : raw ensemble probabilities (0–1)
    params : dict from fit_calibrator() or load_calibrator()

    Returns
    -------
    (cal_home, cal_draw, cal_away) normalised tuple
    """
    def _cal(p: float, ab: list) -> float:
        a, b = ab
        return 1.0 / (1.0 + math.exp(max(-500.0, min(500.0, a * p + b))))

    ch = _cal(prob_home, params["home"])
    cd = _cal(prob_draw, params["draw"])
    ca = _cal(prob_away, params["away"])

    total = ch + cd + ca
    if total < 1e-9:
        return prob_home, prob_draw, prob_away

    return ch / total, cd / total, ca / total


def save_calibrator(params: dict, path: str = _DEFAULT_CALIB_PATH) -> None:
    """Persist calibration parameters as JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2)


def load_calibrator(path: str = _DEFAULT_CALIB_PATH) -> dict | None:
    """Load calibration parameters. Returns None if the file doesn't exist."""
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None
