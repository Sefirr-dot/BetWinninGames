"""
Draw probability classifier — logistic regression trained on resolved live picks.

Learns P(draw) from sub-model draw probabilities available at prediction time.
Replaces the hand-tuned draw nudge in ensemble.py when enough data is available.

Features
--------
  dc_draw  : Dixon-Coles predicted draw prob
  elo_draw : Elo predicted draw prob
  h2h_draw : H2H draw prob × has_sufficient_h2h  (0.0 when insufficient)
  mkt_draw : market implied draw prob             (0.0 when no odds)

Why logistic regression?  With 50–500 live picks the model must not overfit.
scipy.special.expit + L-BFGS-B give a calibrated Platt-style sigmoid with zero
extra dependencies beyond what is already in requirements.txt.

Training
--------
    python -c "from algorithms.draw_model import train; print(train('cache/picks_history.db'))"

The model is retrained automatically at the end of each tracker.py run when
>= DRAW_MODEL_MIN_SAMPLES resolved live picks are available.

Cache
-----
    cache/draw_model.json  —  {"weights": [bias, w_dc, w_elo, w_h2h, w_mkt], "n_samples": N}
"""

import json
import os
import sqlite3

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

DRAW_MODEL_PATH = "cache/draw_model.json"
DRAW_MODEL_MIN_SAMPLES = 50   # fewer than meta_learner because simpler model


# ---------------------------------------------------------------------------
# Feature builder
# ---------------------------------------------------------------------------

def _fv(dc_draw: float, elo_draw: float, h2h_draw: float, mkt_draw: float) -> np.ndarray:
    """Feature vector with bias term. All inputs in [0, 1]."""
    return np.array([1.0, dc_draw, elo_draw, h2h_draw, mkt_draw], dtype=float)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _log_loss(weights: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
    probs = np.clip(expit(X @ weights), 1e-7, 1 - 1e-7)
    return -float(np.mean(y * np.log(probs) + (1 - y) * np.log(1 - probs)))


def train(db_path: str) -> str:
    """
    Fit logistic regression on resolved live picks.

    Returns a status string suitable for logging.
    """
    if not os.path.exists(db_path):
        return f"draw_model: DB not found ({db_path})"

    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT sub_preds, actual_result
            FROM   picks
            WHERE  source = 'live'
              AND  actual_result IS NOT NULL
              AND  sub_preds IS NOT NULL
            """
        ).fetchall()

    if len(rows) < DRAW_MODEL_MIN_SAMPLES:
        return (
            f"draw_model: solo {len(rows)} picks resueltos "
            f"(necesita >= {DRAW_MODEL_MIN_SAMPLES})"
        )

    X_rows, y_rows = [], []
    for sub_preds_json, actual_result in rows:
        try:
            sp       = json.loads(sub_preds_json)
            dc_draw  = float((sp.get("dc") or {}).get("prob_draw") or 0.0)
            elo_draw = float((sp.get("elo") or {}).get("prob_draw") or 0.0)
            h2h      = sp.get("h2h") or {}
            h2h_draw = float(h2h.get("prob_draw") or 0.0) if h2h.get("sufficient") else 0.0
            ctx      = sp.get("context") or {}
            mkt_draw = float(ctx.get("mkt_px") or 0.0)
            X_rows.append(_fv(dc_draw, elo_draw, h2h_draw, mkt_draw))
            y_rows.append(1.0 if actual_result == "D" else 0.0)
        except (json.JSONDecodeError, TypeError, ValueError):
            continue

    if len(X_rows) < DRAW_MODEL_MIN_SAMPLES:
        return f"draw_model: solo {len(X_rows)} filas válidas tras parsear sub_preds"

    X = np.array(X_rows)
    y = np.array(y_rows)

    result = minimize(_log_loss, np.zeros(X.shape[1]), args=(X, y), method="L-BFGS-B")

    os.makedirs("cache", exist_ok=True)
    with open(DRAW_MODEL_PATH, "w", encoding="utf-8") as f:
        json.dump({"weights": result.x.tolist(), "n_samples": len(X_rows),
                   "source": "live"}, f)

    return (
        f"draw_model: {len(X_rows)} picks | "
        f"tasa_draw={y.mean():.3f} | loss={result.fun:.4f} | "
        f"pesos={[round(w, 4) for w in result.x.tolist()]}"
    )


# ---------------------------------------------------------------------------
# Backtest pre-training
# ---------------------------------------------------------------------------

def pretrain_from_backtest(backtest_results: list[dict]) -> str:
    """
    Pre-train draw model on walk-forward backtest results.

    This gives the model a calibrated prior before any live picks accumulate.
    Distribution shift risk is low because features are relative (prob_draw
    ratios between sub-models, not absolute levels) and draw rates are stable.

    When live picks later reach DRAW_MODEL_MIN_SAMPLES, train() overwrites this
    with live-data weights — live data always takes priority.

    Parameters
    ----------
    backtest_results : list of dicts from backtest.run_backtest()
        Each dict must have keys "prediction" and "actual".

    Returns
    -------
    Status string for logging.
    """
    if not backtest_results:
        return "draw_model pretrain: sin resultados de backtest"

    # Never overwrite a model already trained on live data
    try:
        with open(DRAW_MODEL_PATH, encoding="utf-8") as _f:
            if json.load(_f).get("source") == "live":
                return "draw_model pretrain: omitido (ya existe modelo live)"
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        pass

    X_rows, y_rows = [], []
    for r in backtest_results:
        try:
            pred     = r["prediction"]
            actual   = r["actual"]
            dc_draw  = float((pred.get("dc") or {}).get("prob_draw") or 0.0)
            elo_draw = float((pred.get("elo") or {}).get("prob_draw") or 0.0)
            h2h      = pred.get("h2h") or {}
            h2h_draw = float(h2h.get("prob_draw") or 0.0) if h2h.get("sufficient") else 0.0
            ctx      = pred.get("_context") or {}
            # mkt_px from context if available; else derive from market_odds
            mkt_draw = float(ctx.get("mkt_px") or 0.0)
            if not mkt_draw:
                mkt = r.get("market_odds")
                if mkt:
                    try:
                        r1 = 1 / mkt["odds_1"]; rx = 1 / mkt["odds_x"]; r2 = 1 / mkt["odds_2"]
                        s = r1 + rx + r2
                        mkt_draw = rx / s if s > 0 else 0.0
                    except (KeyError, ZeroDivisionError):
                        mkt_draw = 0.0
            X_rows.append(_fv(dc_draw, elo_draw, h2h_draw, mkt_draw))
            y_rows.append(1.0 if actual == "draw" else 0.0)
        except (TypeError, ValueError):
            continue

    if len(X_rows) < DRAW_MODEL_MIN_SAMPLES:
        return f"draw_model pretrain: solo {len(X_rows)} filas válidas"

    X = np.array(X_rows)
    y = np.array(y_rows)

    result = minimize(_log_loss, np.zeros(X.shape[1]), args=(X, y), method="L-BFGS-B")

    os.makedirs("cache", exist_ok=True)
    with open(DRAW_MODEL_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "weights":   result.x.tolist(),
            "n_samples": len(X_rows),
            "source":    "backtest_pretrain",
        }, f)

    return (
        f"draw_model pretrain (backtest): {len(X_rows)} partidos | "
        f"tasa_draw={y.mean():.3f} | loss={result.fun:.4f} | "
        f"pesos={[round(w, 4) for w in result.x.tolist()]}"
    )


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def load_model() -> list[float] | None:
    """Load weights. Returns None when file absent or corrupt."""
    try:
        with open(DRAW_MODEL_PATH, encoding="utf-8") as f:
            return json.load(f)["weights"]
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        return None


def predict(
    dc_draw: float,
    elo_draw: float,
    h2h_draw: float,
    mkt_draw: float,
    weights: list[float],
) -> float:
    """
    Calibrated draw probability.

    Parameters
    ----------
    dc_draw, elo_draw : sub-model draw probs (0..1)
    h2h_draw          : H2H draw prob (0.0 when insufficient)
    mkt_draw          : market implied draw prob (0.0 when no odds)
    weights           : from load_model()

    Returns
    -------
    float in [0.05, 0.45]
    """
    prob = float(expit(np.dot(_fv(dc_draw, elo_draw, h2h_draw, mkt_draw),
                              np.array(weights))))
    return max(0.05, min(0.45, prob))
