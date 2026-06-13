"""
DC/Elo weight sensitivity analysis.

Replicates exactly what weight_optimizer.py optimises — the *normalised DC/Elo
blend* scored on 1X2 outcomes — but sweeps the full weight range and evaluates
it on every resolved pick with stored sub_preds, instead of only the ~85 picks
that carry a Pinnacle closing line.

Purpose: check whether the CLV objective (which optimises on the tiny 85-pick
subset) generalises. If the weight that maximises CLV on 85 picks has a poor
Brier score on the full 7920-match sample, the CLV objective is overfitting.

Run:  python scripts/weight_sensitivity.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import db_picks
from config import MODEL_WEIGHTS, PICKS_DB

_IDX = {"home": 0, "draw": 1, "away": 2}


def _load():
    """Return arrays for all resolved picks with usable dc/elo sub_preds."""
    picks = db_picks.get_all_picks(PICKS_DB)
    dc, elo, y, idx, closing, our_impl, is_live, stars = [], [], [], [], [], [], [], []
    for p in picks:
        if p["actual_result"] not in _IDX:
            continue
        raw = p.get("sub_preds")
        if not raw:
            continue
        try:
            sub = json.loads(raw)
        except (TypeError, ValueError):
            continue
        d, e = sub.get("dc"), sub.get("elo")
        if not (d and e):
            continue
        dc.append([d.get("prob_home", 1/3), d.get("prob_draw", 1/3), d.get("prob_away", 1/3)])
        elo.append([e.get("prob_home", 1/3), e.get("prob_draw", 1/3), e.get("prob_away", 1/3)])
        oi = _IDX[p["actual_result"]]
        idx.append(oi)
        y.append([1.0 if i == oi else 0.0 for i in range(3)])
        co = p.get("closing_odds")
        closing.append(float(co) if co and co > 1.0 else np.nan)
        is_live.append(p.get("source") == "live")
        stars.append(p.get("stars") or 0)
    return (np.array(dc), np.array(elo), np.array(y), np.array(idx),
            np.array(closing), np.array(is_live, dtype=bool), np.array(stars))


def _blend(dc, elo, w_dc):
    b = w_dc * dc + (1.0 - w_dc) * elo
    s = b.sum(axis=1, keepdims=True)
    s = np.where(s < 1e-9, 1.0, s)
    return b / s


def _metrics(dc, elo, y, idx, w_dc):
    b = _blend(dc, elo, w_dc)
    brier = float(np.mean(np.sum((b - y) ** 2, axis=1)))
    acc = float(np.mean(np.argmax(b, axis=1) == idx))
    eps = 1e-12
    logloss = float(-np.mean(np.log(np.clip(b[np.arange(len(idx)), idx], eps, 1))))
    return brier, acc, logloss


def _clv(dc, elo, closing, w_dc):
    """Mean CLV vs Pinnacle on the subset that has a closing line (mirrors optimizer)."""
    mask = ~np.isnan(closing)
    if mask.sum() == 0:
        return float("nan")
    b = _blend(dc[mask], elo[mask], w_dc)
    our = b[np.arange(mask.sum()), np.argmax(b, axis=1)]
    return float(np.mean(our - 1.0 / closing[mask]))


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    dc, elo, y, idx, closing, is_live, stars = _load()
    n = len(idx)
    n_clv = int((~np.isnan(closing)).sum())
    print(f"Resolved picks w/ dc+elo sub_preds: {n}  (live={int(is_live.sum())}, with closing line={n_clv})")

    prior_wdc = MODEL_WEIGHTS["dixon_coles"] / (MODEL_WEIGHTS["dixon_coles"] + MODEL_WEIGHTS["elo"])

    grid = np.round(np.arange(0.05, 0.96, 0.05), 2)
    print("\n  w_dc | Brier(all) acc(all) | Brier(live) acc(live) | meanCLV(85)")
    print("  " + "-" * 70)
    rows = []
    for w in grid:
        ba, aa, _ = _metrics(dc, elo, y, idx, w)
        bl, al, _ = _metrics(dc[is_live], elo[is_live], y[is_live], idx[is_live], w) if is_live.sum() else (float("nan"),)*3
        cl = _clv(dc, elo, closing, w)
        rows.append((w, ba, aa, bl, al, cl))
        mark = []
        if abs(w - round(prior_wdc, 2)) < 0.026:
            mark.append("<- prior")
        if abs(w - 0.10) < 0.026:
            mark.append("<- raw CLV opt")
        if abs(w - 0.60) < 0.026:
            mark.append("<- shrunk (saved)")
        print(f"  {w:.2f} | {ba:9.4f} {aa:7.3f} | {bl:10.4f} {al:8.3f} | {cl:+.4f}  {' '.join(mark)}")

    rows = np.array([r[:6] for r in rows])
    best_all = grid[int(np.argmin(rows[:, 1]))]
    best_live = grid[int(np.nanargmin(rows[:, 3]))]
    best_clv = grid[int(np.nanargmax(rows[:, 5]))]
    print("\n  Brier-optimal w_dc on full 7920 :", best_all)
    print("  Brier-optimal w_dc on live only  :", best_live)
    print("  CLV-optimal   w_dc (85 picks)    :", best_clv,
          "  <-- what weight_optimizer chases")
    print(f"  Config prior  w_dc               : {prior_wdc:.3f}")


if __name__ == "__main__":
    main()
