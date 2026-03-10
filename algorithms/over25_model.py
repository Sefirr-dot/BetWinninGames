"""
Over 2.5 goals probability calibrator — logistic regression.

The Monte Carlo simulation gives a raw P(over25) from DC lambda/mu.  This
module learns a calibration layer on top of that raw estimate, correcting
for systematic biases (e.g. low-scoring leagues, defensive setups).

Features
--------
  mc_over25   : Monte Carlo predicted over25 prob  (main predictor)
  lam_plus_mu : expected total goals (λ + μ)       (structural signal)
  btts_prob   : BTTS probability                    (correlated with goals)

Why logistic regression?  Same reasoning as draw_model: few samples,
interpretable, no new dependencies.

Training
--------
    python -c "from algorithms.over25_model import train; print(train('cache/picks_history.db'))"

Auto-retrained by tracker.py when >= OVER25_MODEL_MIN_SAMPLES resolved picks exist.
Pre-trained automatically at end of backtest.py run.

Cache
-----
    cache/over25_model.json  —  {"weights": [bias, w_mc, w_lam_mu, w_btts],
                                  "n_samples": N, "source": "live"|"backtest_pretrain"}
"""

import json
import os
import sqlite3

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

OVER25_MODEL_PATH = "cache/over25_model.json"
OVER25_MODEL_MIN_SAMPLES = 50


def _fv(mc_over25: float, lam_plus_mu: float, btts_prob: float) -> np.ndarray:
    """Feature vector with bias term."""
    return np.array([1.0, mc_over25, lam_plus_mu, btts_prob], dtype=float)


_L2 = 0.01  # L2 regularisation — prevents unstable convergence across backtest runs


def _log_loss(weights: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
    probs = np.clip(expit(X @ weights), 1e-7, 1 - 1e-7)
    nll   = -float(np.mean(y * np.log(probs) + (1 - y) * np.log(1 - probs)))
    l2    = _L2 * float(np.sum(weights[1:] ** 2))  # skip bias term
    return nll + l2


# ---------------------------------------------------------------------------
# Training on live picks
# ---------------------------------------------------------------------------

def train(db_path: str) -> str:
    """Fit on resolved live picks. Returns status string."""
    if not os.path.exists(db_path):
        return f"over25_model: DB not found ({db_path})"

    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT over25, btts, sub_preds, actual_over25
            FROM   picks
            WHERE  source = 'live'
              AND  actual_over25 IS NOT NULL
              AND  over25 IS NOT NULL
            """
        ).fetchall()

    if len(rows) < OVER25_MODEL_MIN_SAMPLES:
        return (
            f"over25_model: solo {len(rows)} picks resueltos "
            f"(necesita >= {OVER25_MODEL_MIN_SAMPLES})"
        )

    X_rows, y_rows = [], []
    for mc_over25, btts_prob, sub_preds_json, actual_over25 in rows:
        try:
            sp  = json.loads(sub_preds_json) if sub_preds_json else {}
            dc  = sp.get("dc") or {}
            lam = float(dc.get("lambda_") or 1.3)
            mu  = float(dc.get("mu_") or 1.0)
            X_rows.append(_fv(float(mc_over25 or 0.5), lam + mu, float(btts_prob or 0.5)))
            y_rows.append(1.0 if actual_over25 else 0.0)
        except (json.JSONDecodeError, TypeError, ValueError):
            continue

    if len(X_rows) < OVER25_MODEL_MIN_SAMPLES:
        return f"over25_model: solo {len(X_rows)} filas válidas tras parsear"

    X, y = np.array(X_rows), np.array(y_rows)
    result = minimize(_log_loss, np.zeros(X.shape[1]), args=(X, y), method="L-BFGS-B")

    os.makedirs("cache", exist_ok=True)
    with open(OVER25_MODEL_PATH, "w", encoding="utf-8") as f:
        json.dump({"weights": result.x.tolist(), "n_samples": len(X_rows),
                   "source": "live"}, f)

    return (
        f"over25_model: {len(X_rows)} picks | "
        f"tasa_over25={y.mean():.3f} | loss={result.fun:.4f} | "
        f"pesos={[round(w, 4) for w in result.x.tolist()]}"
    )


# ---------------------------------------------------------------------------
# Backtest pre-training
# ---------------------------------------------------------------------------

def pretrain_from_backtest(backtest_results: list[dict]) -> str:
    """
    Pre-train on walk-forward backtest results.

    Skips automatically if a live-trained model already exists.
    """
    if not backtest_results:
        return "over25_model pretrain: sin resultados"

    try:
        with open(OVER25_MODEL_PATH, encoding="utf-8") as _f:
            if json.load(_f).get("source") == "live":
                return "over25_model pretrain: omitido (ya existe modelo live)"
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        pass

    X_rows, y_rows = [], []
    for r in backtest_results:
        try:
            pred       = r["prediction"]
            mc_over25  = float(pred.get("over25") or 0.5)
            btts_prob  = float(pred.get("btts_prob") or 0.5)
            lam        = float(pred.get("expected_goals_home") or 1.3)
            mu         = float(pred.get("expected_goals_away") or 1.0)
            actual     = r["actual_over25"]
            X_rows.append(_fv(mc_over25, lam + mu, btts_prob))
            y_rows.append(1.0 if actual else 0.0)
        except (TypeError, ValueError):
            continue

    if len(X_rows) < OVER25_MODEL_MIN_SAMPLES:
        return f"over25_model pretrain: solo {len(X_rows)} filas válidas"

    X, y = np.array(X_rows), np.array(y_rows)
    result = minimize(_log_loss, np.zeros(X.shape[1]), args=(X, y), method="L-BFGS-B")

    os.makedirs("cache", exist_ok=True)
    with open(OVER25_MODEL_PATH, "w", encoding="utf-8") as f:
        json.dump({"weights": result.x.tolist(), "n_samples": len(X_rows),
                   "source": "backtest_pretrain"}, f)

    return (
        f"over25_model pretrain (backtest): {len(X_rows)} partidos | "
        f"tasa_over25={y.mean():.3f} | loss={result.fun:.4f} | "
        f"pesos={[round(w, 4) for w in result.x.tolist()]}"
    )


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def load_model() -> list[float] | None:
    """Load weights. Returns None when absent or corrupt."""
    try:
        with open(OVER25_MODEL_PATH, encoding="utf-8") as f:
            return json.load(f)["weights"]
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        return None


def predict(
    mc_over25: float,
    lam_plus_mu: float,
    btts_prob: float,
    weights: list[float],
) -> float:
    """
    Calibrated Over 2.5 probability.

    Returns float in [0.05, 0.95].
    """
    prob = float(expit(np.dot(_fv(mc_over25, lam_plus_mu, btts_prob),
                              np.array(weights))))
    return max(0.05, min(0.95, prob))
