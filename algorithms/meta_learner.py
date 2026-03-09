"""
XGBoost meta-learner for BetWinninGames.

Trains a gradient-boosted classifier on resolved picks from picks_history.db,
using the four sub-model probabilities (DC, Elo, Form, H2H) as features.

When active, replaces the fixed weighted blend in ensemble.py with
data-driven predictions.  Falls back to the weighted blend automatically
when the model file is absent or when features cannot be built.

Requirements
------------
    pip install xgboost>=1.7

Cache
-----
    cache/meta_learner.pkl  — pickle of {"model": XGBClassifier, "n_samples": int, ...}

Features (25)
-------------
    dc_ph, dc_pd, dc_pa          Dixon-Coles 1X2
    elo_ph, elo_pd, elo_pa       Elo 1X2
    form_ph, form_pd, form_pa    Form 1X2
    h2h_ph, h2h_pd, h2h_pa       H2H 1X2 (1/3 each when insufficient)
    h2h_sufficient               0/1 flag
    dc_lambda                    Expected home goals (DC)
    dc_mu                        Expected away goals (DC)
    dc_over25                    DC probability over 2.5 goals
    elo_home_norm                Home Elo / 1500 (normalised)
    elo_away_norm                Away Elo / 1500 (normalised)
    xg_form_home                 xG-based form strength, home team (0..1)
    xg_form_away                 xG-based form strength, away team (0..1)
    pos_diff_norm                (away_pos - home_pos) / 19, positive = home ranks higher
    fatigue_diff                 (home_days - away_days) / 7, positive = home more rested
    mkt_ph                       Market implied prob home (margin-removed), 1/3 when no odds
    mkt_px                       Market implied prob draw (margin-removed), 1/3 when no odds
    mkt_pa                       Market implied prob away (margin-removed), 1/3 when no odds

Target classes: 0=home  1=draw  2=away
"""

import json
import os
import pickle

import numpy as np

try:
    import xgboost as xgb
    _XGB_AVAILABLE = True
except ImportError:
    xgb = None
    _XGB_AVAILABLE = False

_MODEL_PATH  = "cache/meta_learner.pkl"
_MIN_SAMPLES = 200

FEATURE_NAMES = [
    "dc_ph", "dc_pd", "dc_pa",
    "elo_ph", "elo_pd", "elo_pa",
    "form_ph", "form_pd", "form_pa",
    "h2h_ph", "h2h_pd", "h2h_pa",
    "h2h_sufficient",
    "dc_lambda", "dc_mu", "dc_over25",
    "elo_home_norm", "elo_away_norm",
    # Context features (Fase B)
    "xg_form_home",   # xG-based form strength, home team (0..1)
    "xg_form_away",   # xG-based form strength, away team (0..1)
    "pos_diff_norm",  # (away_pos - home_pos) / 19, positive = home ranks higher
    "fatigue_diff",   # (home_days - away_days) / 7, positive = home more rested
    # Market features (Fase C — bookmaker implied probs, margin-removed)
    "mkt_ph",         # market implied prob home (1/3 when no odds)
    "mkt_px",         # market implied prob draw (1/3 when no odds)
    "mkt_pa",         # market implied prob away (1/3 when no odds)
]

_LABEL = {"home": 0, "draw": 1, "away": 2}
_LABEL_INV = {v: k for k, v in _LABEL.items()}


# ---------------------------------------------------------------------------
# Feature builder
# ---------------------------------------------------------------------------

def _build_features(sub_preds: dict) -> list[float] | None:
    """
    Build a 13-float feature vector from a sub_preds dict.

    Accepts both formats:
      - DB format (simplified):  {"dc": {prob_home, prob_draw, prob_away}, ...}
      - Live format (full dict): {"dc": {prob_home, ..., lambda_, ...}, ...}
    """
    dc   = sub_preds.get("dc")   or {}
    elo  = sub_preds.get("elo")  or {}
    form = sub_preds.get("form") or {}
    h2h  = sub_preds.get("h2h") or {}

    # Require all three main models
    if not (dc.get("prob_home") and elo.get("prob_home") and form.get("prob_home")):
        return None

    h2h_suf = 1.0 if h2h.get("sufficient") else 0.0
    h2h_ph  = h2h.get("prob_home", 1 / 3) if h2h_suf else 1 / 3
    h2h_pd  = h2h.get("prob_draw", 1 / 3) if h2h_suf else 1 / 3
    h2h_pa  = h2h.get("prob_away", 1 / 3) if h2h_suf else 1 / 3

    # Extra features — fall back to league-average values for old DB records
    dc_lambda    = dc.get("lambda_", 1.3)
    dc_mu        = dc.get("mu_",     1.0)
    dc_over25    = dc.get("over25",  0.5)
    elo_home_raw = elo.get("rating_home", 1500.0)
    elo_away_raw = elo.get("rating_away", 1500.0)

    ctx = sub_preds.get("context") or {}
    xg_form_home = ctx.get("xg_form_home", 0.50)
    xg_form_away = ctx.get("xg_form_away", 0.50)
    pos_diff_norm = ctx.get("pos_diff_norm", 0.0)
    fatigue_diff  = ctx.get("fatigue_diff",  0.0)
    mkt_ph = ctx.get("mkt_ph", 1 / 3)
    mkt_px = ctx.get("mkt_px", 1 / 3)
    mkt_pa = ctx.get("mkt_pa", 1 / 3)

    return [
        dc.get("prob_home",   1 / 3), dc.get("prob_draw",   1 / 3), dc.get("prob_away",   1 / 3),
        elo.get("prob_home",  1 / 3), elo.get("prob_draw",  1 / 3), elo.get("prob_away",  1 / 3),
        form.get("prob_home", 1 / 3), form.get("prob_draw", 1 / 3), form.get("prob_away", 1 / 3),
        h2h_ph, h2h_pd, h2h_pa,
        h2h_suf,
        dc_lambda, dc_mu, dc_over25,
        elo_home_raw / 1500.0, elo_away_raw / 1500.0,
        xg_form_home, xg_form_away, pos_diff_norm, fatigue_diff,
        mkt_ph, mkt_px, mkt_pa,
    ]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(db_path: str, output_path: str = _MODEL_PATH, real_only: bool = True) -> dict:
    """
    Train the XGBoost meta-learner from resolved picks in picks_history.db.

    Parameters
    ----------
    db_path     : path to picks_history.db
    output_path : where to save the trained model (pickle)
    real_only   : if True (default), only use picks from main.py (source='live').
                  Backtest-seeded picks (source='backtest') are excluded because
                  they come from different seasons and cause distribution shift.

    Returns
    -------
    dict with "n_samples", "brier_train", "brier_val", "output_path"
    — or {"error": "..."}
    """
    if not _XGB_AVAILABLE:
        return {"error": "xgboost no instalado. Ejecuta: pip install xgboost>=1.7"}

    import db_picks

    picks = db_picks.get_real_picks(db_path) if real_only else db_picks.get_all_picks(db_path)

    X, y = [], []
    for p in picks:
        outcome = p.get("actual_result")
        if outcome not in _LABEL:
            continue
        raw = p.get("sub_preds")
        if not raw:
            continue
        try:
            sub = json.loads(raw)
        except Exception:
            continue
        feats = _build_features(sub)
        if feats is None:
            continue
        X.append(feats)
        y.append(_LABEL[outcome])

    n = len(X)
    if n < _MIN_SAMPLES:
        return {"error": f"Muestras insuficientes: {n}/{_MIN_SAMPLES}. Ejecuta más backtests con --seed-db."}

    X_arr = np.array(X, dtype=np.float32)
    y_arr = np.array(y, dtype=np.int32)

    # 80/20 stratified-ish split (shuffle then cut)
    rng      = np.random.default_rng(42)
    idx      = rng.permutation(n)
    n_val    = max(1, n // 5)
    val_idx  = idx[:n_val]
    train_idx = idx[n_val:]

    X_train, X_val = X_arr[train_idx], X_arr[val_idx]
    y_train, y_val = y_arr[train_idx], y_arr[val_idx]

    # Native XGBoost API — no scikit-learn required
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=FEATURE_NAMES)
    dval   = xgb.DMatrix(X_val,   label=y_val,   feature_names=FEATURE_NAMES)

    params = {
        "objective":        "multi:softprob",
        "num_class":        3,
        "max_depth":        4,
        "eta":              0.05,
        "subsample":        0.8,
        "colsample_bytree": 0.8,
        "eval_metric":      "mlogloss",
        "seed":             42,
        "verbosity":        0,
    }

    # Early stopping: stop when val mlogloss doesn't improve for 25 rounds.
    # xgb.callback.EarlyStopping available since XGBoost 1.3 (we require >=1.7).
    evals_result = {}
    try:
        callbacks = [xgb.callback.EarlyStopping(rounds=25, save_best=True, maximize=False)]
        model = xgb.train(
            params, dtrain,
            num_boost_round=600,
            evals=[(dtrain, "train"), (dval, "val")],
            callbacks=callbacks,
            evals_result=evals_result,
        )
    except Exception:
        # Fallback without early stopping (very old XGBoost or env issue)
        model = xgb.train(params, dtrain, num_boost_round=200)

    best_round = int(getattr(model, "best_iteration", 200))

    # Brier score on training and validation sets
    d_full = xgb.DMatrix(X_arr, label=y_arr, feature_names=FEATURE_NAMES)
    proba_train = model.predict(dtrain).reshape(-1, 3)
    proba_val   = model.predict(dval).reshape(-1, 3)

    def _brier(proba_mat, labels):
        n_s = len(labels)
        y_oh = np.zeros((n_s, 3), dtype=np.float32)
        y_oh[np.arange(n_s), labels] = 1.0
        return float(np.mean(np.sum((proba_mat - y_oh) ** 2, axis=1)))

    brier_train = _brier(proba_train, y_train)
    brier_val   = _brier(proba_val,   y_val)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump({
            "model":       model,
            "n_samples":   n,
            "n_train":     len(train_idx),
            "n_val":       len(val_idx),
            "best_round":  best_round,
            "brier_train": round(brier_train, 4),
            "brier_val":   round(brier_val,   4),
        }, f)

    return {
        "n_samples":   n,
        "n_train":     len(train_idx),
        "n_val":       len(val_idx),
        "best_round":  best_round,
        "brier_train": round(brier_train, 4),
        "brier_val":   round(brier_val,   4),
        "output_path": output_path,
    }


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def load_model(path: str = _MODEL_PATH):
    """Load trained model from disk. Returns None if unavailable."""
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
        model = data.get("model")
        if model is not None and hasattr(model, "num_features") and model.num_features() != len(FEATURE_NAMES):
            print(f"  [meta_learner] Modelo obsoleto ({model.num_features()} features vs "
                  f"{len(FEATURE_NAMES)} esperadas) — se reentrenara en el proximo tracker run.")
            return None
        return model
    except Exception:
        return None


def predict(sub_preds: dict, model) -> tuple[float, float, float] | None:
    """
    Predict 1X2 probabilities with the meta-learner.

    Returns (prob_home, prob_draw, prob_away) or None if features unavailable.
    """
    if model is None or not _XGB_AVAILABLE:
        return None
    feats = _build_features(sub_preds)
    if feats is None:
        return None
    X = np.array([feats], dtype=np.float32)
    dtest = xgb.DMatrix(X, feature_names=FEATURE_NAMES)
    proba = model.predict(dtest).reshape(-1, 3)[0]   # [home, draw, away]
    return float(proba[0]), float(proba[1]), float(proba[2])
