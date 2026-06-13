"""
Informe de verificacion de datos — comprueba de punta a punta que el sistema
trabaja con datos reales y frescos.

Uso:
    python scripts/verify_data.py

Imprime:
  1. Cobertura por liga/temporada en cache/football_data.db (partidos + ultima fecha)
  2. Cobertura xG en cache/understat_xg.db
  3. Cobertura fdco en cache/fdco_data.db
  4. Estado de cada modelo entrenado (source, n_samples, fecha de archivo)
  5. picks_history.db: conteos por source, pendientes, duplicados (debe ser 0)
  6. 5 picks live resueltos al azar — prediccion vs resultado real (verificables
     contra cualquier web de resultados)
"""

import json
import os
import sqlite3
import sys
from datetime import datetime

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

# Partidos esperados por temporada completa
EXPECTED = {"PL": 380, "PD": 380, "BL1": 306, "FL1": 306}
COMP_IDS = {"PL": 2021, "PD": 2014, "BL1": 2002, "FL1": 2015}


def _section(title: str) -> None:
    print(f"\n{'=' * 64}\n  {title}\n{'=' * 64}")


def check_football_data() -> None:
    _section("1. football-data.org (cache/football_data.db)")
    db = "cache/football_data.db"
    if not os.path.exists(db):
        print("  [!] no existe")
        return
    conn = sqlite3.connect(db)
    rows = conn.execute("SELECT key, value FROM cache WHERE key LIKE 'season_%'").fetchall()
    by_key = {}
    for key, value in rows:
        try:
            matches = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(matches, list):
            continue
        finished = [m for m in matches if m.get("status") == "FINISHED"]
        last = max((m.get("utcDate", "") for m in finished), default="-")
        by_key[key] = (len(matches), len(finished), last[:10])
    code_by_comp = {v: k for k, v in COMP_IDS.items()}
    for key in sorted(by_key):
        n, n_fin, last = by_key[key]
        parts = key.split("_")  # season_{comp}_{year}
        comp, year = int(parts[1]), parts[2]
        league = code_by_comp.get(comp, str(comp))
        exp = EXPECTED.get(league, "?")
        flag = "OK " if n_fin == exp else "!! "
        print(f"  [{flag}] {league} {year}: {n_fin}/{exp} finalizados | ultimo: {last}")
    conn.close()


def check_understat() -> None:
    _section("2. xG Understat (cache/understat_xg.db)")
    db = "cache/understat_xg.db"
    if not os.path.exists(db):
        print("  [!] no existe")
        return
    conn = sqlite3.connect(db)
    try:
        rows = conn.execute(
            "SELECT league, season, COUNT(*), MAX(match_date) FROM xg_matches GROUP BY league, season"
        ).fetchall()
        for league, season, n, last in sorted(rows):
            print(f"  {league} {season}: {n} partidos con xG | ultimo: {str(last)[:10]}")
    except sqlite3.OperationalError as exc:
        print(f"  [!] error: {exc}")
    conn.close()


def check_fdco() -> None:
    _section("3. football-data.co.uk (cache/fdco_data.db)")
    db = "cache/fdco_data.db"
    if not os.path.exists(db):
        print("  [-] no existe (se descarga al correr backtest)")
        return
    conn = sqlite3.connect(db)
    try:
        rows = conn.execute(
            "SELECT league, season, COUNT(*), MAX(match_date) FROM fdco_matches GROUP BY league, season"
        ).fetchall()
        for league, season, n, last in sorted(rows):
            print(f"  {league} {season}: {n} partidos | ultimo: {str(last)[:10]}")
    except sqlite3.OperationalError as exc:
        print(f"  [!] error: {exc}")
    conn.close()


def check_models() -> None:
    _section("4. Modelos entrenados (cache/)")
    for path in ("cache/draw_model.json", "cache/over25_model.json",
                 "cache/model_weights.json", "cache/calibrator.json"):
        if not os.path.exists(path):
            print(f"  [-] {os.path.basename(path)}: ausente")
            continue
        mtime = datetime.fromtimestamp(os.path.getmtime(path)).strftime("%Y-%m-%d %H:%M")
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            src = data.get("source", "?")
            n = data.get("n_samples", data.get("_optimised_from_n", "?"))
            print(f"  [OK] {os.path.basename(path)}: source={src} n={n} mtime={mtime}")
        except (json.JSONDecodeError, OSError) as exc:
            print(f"  [!] {os.path.basename(path)}: ilegible ({exc})")
    for path in ("cache/meta_learner.pkl",):
        status = "presente" if os.path.exists(path) else "ausente"
        print(f"  [-] {os.path.basename(path)}: {status}")


def check_picks() -> None:
    _section("5. picks_history.db")
    db = "cache/picks_history.db"
    if not os.path.exists(db):
        print("  [!] no existe")
        return
    conn = sqlite3.connect(db)
    cur = conn.cursor()
    for src, n in cur.execute("SELECT source, COUNT(*) FROM picks GROUP BY source"):
        print(f"  source={src}: {n} picks")
    n_pending = cur.execute(
        "SELECT COUNT(*) FROM picks WHERE source='live' AND actual_result IS NULL"
    ).fetchone()[0]
    print(f"  picks live sin resolver: {n_pending}")
    n_dup = cur.execute(
        """SELECT COALESCE(SUM(c - 1), 0) FROM (
               SELECT COUNT(*) c FROM picks WHERE source='backtest'
               GROUP BY match_date, home_team, away_team HAVING c > 1)"""
    ).fetchone()[0]
    flag = "OK" if n_dup == 0 else "!!"
    print(f"  [{flag}] filas backtest duplicadas: {n_dup} (debe ser 0)")
    row = cur.execute(
        "SELECT COUNT(*), MAX(match_date) FROM picks WHERE source='live' AND actual_result IS NOT NULL"
    ).fetchone()
    print(f"  picks live resueltos: {row[0]} | ultimo partido: {row[1]}")
    acc = cur.execute(
        """SELECT stars, COUNT(*), SUM(best_outcome = actual_result)
           FROM picks WHERE source='live' AND actual_result IS NOT NULL
           GROUP BY stars ORDER BY stars"""
    ).fetchall()
    for stars, n, hits in acc:
        print(f"    {stars}*: {hits}/{n} ({100.0 * hits / n:.0f}%)")
    conn.close()


def sample_picks() -> None:
    _section("6. Muestra verificable: 5 picks live resueltos al azar")
    conn = sqlite3.connect("cache/picks_history.db")
    rows = conn.execute(
        """SELECT match_date, home_team, away_team, league, best_outcome,
                  ROUND(best_prob * 100, 1), actual_result
           FROM picks WHERE source='live' AND actual_result IS NOT NULL
           ORDER BY RANDOM() LIMIT 5"""
    ).fetchall()
    for d, h, a, lg, pred, prob, actual in rows:
        hit = "ACIERTO" if pred == actual else "FALLO  "
        print(f"  [{hit}] {d} {lg}: {h} vs {a} | pred={pred} ({prob}%) | real={actual}")
    conn.close()
    print("\n  Verifica cualquiera de estos partidos en flashscore.com / google.")


if __name__ == "__main__":
    print(f"BetWinninGames — informe de verificacion de datos")
    print(f"Generado: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    check_football_data()
    check_understat()
    check_fdco()
    check_models()
    check_picks()
    sample_picks()
    print()
