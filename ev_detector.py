"""
+EV detector (prototype) — Pinnacle-as-truth, soft-book edge.

Instead of using the in-house ensemble as the source of "true" probability
(which the backtest showed cannot beat the closing line), this clones the
value_detector logic but treats Pinnacle's margin-removed line as truth.

For each match:
  truth_prob(outcome) = Pinnacle no-vig implied prob   (get_pinnacle_implied)
  best_soft_odds      = best price across all books     (odds/YYYY-MM-DD.csv)
  EV per 1u stake     = truth_prob * best_odds - 1

A bet is +EV when EV >= EV_THRESHOLD. Rows whose best price is Pinnacle itself
are skipped (no soft edge — you'd be betting Pinnacle vs Pinnacle).

This reuses the EXISTING odds CSVs and Pinnacle snapshots — no new API calls.
It is a viability probe, not a betting system: it answers "given the data we
already collect, how many +EV soft-book opportunities exist, and how big?"

Run:
    python ev_detector.py                  # scan every date with both files
    python ev_detector.py --min-ev 0.02    # only show EV >= 2%
"""

import argparse
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from config import ODDS_DIR
from odds_fetcher import get_pinnacle_implied
from algorithms.value_detector import load_odds_csv

_PINNACLE_DIR = "cache/pinnacle"
_OUTCOMES = (
    ("home", "odds_1", "bk_1"),
    ("draw", "odds_x", "bk_x"),
    ("away", "odds_2", "bk_2"),
)


def _overlapping_dates() -> list[str]:
    """Dates that have BOTH an odds CSV and a Pinnacle snapshot."""
    def _dates(d: str) -> set[str]:
        if not os.path.isdir(d):
            return set()
        return {f[:-4] for f in os.listdir(d) if f.endswith(".csv")}
    return sorted(_dates(ODDS_DIR) & _dates(_PINNACLE_DIR))


def find_ev_bets(date_str: str, min_ev: float = 0.0) -> list[dict]:
    """Return +EV soft-book bets for one date (Pinnacle no-vig as truth)."""
    odds_map = load_odds_csv(date_str)
    if not odds_map:
        return []

    out: list[dict] = []
    for entry in odds_map.values():
        home = entry["home_team"]
        away = entry["away_team"]
        truth = get_pinnacle_implied(home, away, date_str)
        if not truth:
            continue  # no sharp reference for this match

        for outcome, odds_key, bk_key in _OUTCOMES:
            best_odds = entry.get(odds_key, 0.0)
            best_book = (entry.get(bk_key, "") or "").strip()
            if best_odds <= 1.0:
                continue
            # Skip when the best price IS Pinnacle — no soft edge there
            if best_book.lower() == "pinnacle":
                continue

            p_true = truth[outcome]
            if p_true <= 0:
                continue
            fair_odds = 1.0 / p_true
            ev = p_true * best_odds - 1.0          # EV per 1u stake
            if ev >= min_ev:
                out.append({
                    "date":      date_str,
                    "match":     f"{home} vs {away}",
                    "outcome":   outcome,
                    "best_odds": round(best_odds, 2),
                    "best_book": best_book,
                    "fair_odds": round(fair_odds, 2),
                    "p_true":    round(p_true, 4),
                    "ev":        round(ev, 4),
                })
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-ev", type=float, default=0.0,
                    help="minimum EV to flag (e.g. 0.02 = +2%%)")
    args = ap.parse_args()

    dates = _overlapping_dates()
    if not dates:
        print("No hay fechas con odds/ y cache/pinnacle/ simultaneamente.")
        return

    all_bets: list[dict] = []
    n_matches = 0
    for d in dates:
        odds_map = load_odds_csv(d)
        # count matches that have a Pinnacle reference (the scannable universe)
        for e in odds_map.values():
            if get_pinnacle_implied(e["home_team"], e["away_team"], d):
                n_matches += 1
        all_bets.extend(find_ev_bets(d, args.min_ev))

    all_bets.sort(key=lambda b: b["ev"], reverse=True)

    print(f"Fechas escaneadas: {len(dates)}  ({dates[0]} -> {dates[-1]})")
    print(f"Partidos con referencia Pinnacle: {n_matches}")
    print(f"Oportunidades +EV (EV >= {args.min_ev:+.1%}): {len(all_bets)}\n")

    if not all_bets:
        return

    # Summary distribution
    ev_vals = [b["ev"] for b in all_bets]
    print(f"EV medio: {sum(ev_vals)/len(ev_vals):+.2%}   "
          f"max: {max(ev_vals):+.2%}   min: {min(ev_vals):+.2%}\n")

    print(f"{'date':<11} {'outcome':<7} {'odds':>5} {'fair':>5} "
          f"{'EV':>7}  {'book':<12} match")
    print("-" * 90)
    for b in all_bets[:40]:
        print(f"{b['date']:<11} {b['outcome']:<7} {b['best_odds']:>5} "
              f"{b['fair_odds']:>5} {b['ev']:>+7.2%}  "
              f"{b['best_book']:<12} {b['match']}")
    if len(all_bets) > 40:
        print(f"... y {len(all_bets) - 40} mas")


if __name__ == "__main__":
    main()
