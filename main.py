"""
BetWinninGames - Football Prediction Engine

Default behaviour (no flags): automatically detects the current day and
generates predictions for the full Friday→Monday weekend window, saving
one TXT per day.

Usage:
    python main.py                          # auto weekend window
    python main.py --date 2026-03-07        # single specific date
    python main.py --league PL              # filter to one league
    python main.py --no-cache               # ignore cached data
"""

import argparse
import sys
from datetime import date, timedelta, datetime, timezone

# Force UTF-8 stdout so Unicode chars (★, ═, etc.) work on Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import fetcher
import fdco_fetcher
import odds_fetcher
import db_picks
import tracker as tracker_module
import telegram_notifier
import understat_fetcher
from algorithms import dixon_coles, elo as elo_module, ensemble, value_detector
from reporter import generate_js
from config import LEAGUES, API_KEY, PICKS_DB, AI_ADVISOR_ENABLED, AI_ADVISOR_MIN_STARS


def parse_args():
    parser = argparse.ArgumentParser(
        description="BetWinninGames - Football Predictions"
    )
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Single target date YYYY-MM-DD (overrides auto weekend mode)",
    )
    parser.add_argument(
        "--league",
        type=str,
        default=None,
        choices=list(LEAGUES.keys()),
        help="Filter to a single league (default: all 5)",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Ignore cache and re-fetch all data",
    )
    return parser.parse_args()


def weekend_dates() -> list[date]:
    """
    Return [Friday, Saturday, Sunday, Monday] for the current or next weekend.

    Rules:
      - Tue / Wed / Thu  → upcoming Fri-Mon
      - Fri / Sat / Sun  → this Fri-Mon (Fri of current week)
      - Mon              → today is still part of the window, return Mon only
                          (the weekend just ended, no point showing last Fri-Sun)
    """
    today = date.today()
    wd = today.weekday()  # 0=Mon … 4=Fri, 5=Sat, 6=Sun

    if wd <= 3:
        # Mon–Thu – jump to the coming Friday
        # Tue-Thu – jump to the coming Friday
        friday = today + timedelta(days=(4 - wd))
    elif wd == 4:
        friday = today
    elif wd == 5:
        friday = today - timedelta(days=1)
    else:  # Sunday
        friday = today - timedelta(days=2)

    return [friday + timedelta(days=i) for i in range(4)]  # Fri, Sat, Sun, Mon


def predict_for_date(
    target_date: str,
    historical: list[dict],
    dc_params: dict,
    elo_ratings: dict,
    league: str | None,
    standings_map: dict | None = None,
    elo_home_ratings: dict | None = None,
    elo_away_ratings: dict | None = None,
) -> list[dict]:
    """Fetch matches for one date and run the ensemble. Returns ranked predictions."""
    matches = fetcher.get_matches_for_date(target_date, league)
    if not matches:
        return []

    ref_date = datetime.strptime(target_date, "%Y-%m-%d").date()

    # Load odds CSV (already downloaded by fetch_window) for per-match market blend
    odds_map = value_detector.load_odds_csv(target_date)

    # Compute CSV age for stale-odds detection in ensemble
    import time as _time, os as _os
    from config import ODDS_DIR as _ODDS_DIR
    _csv_path = _os.path.join(_ODDS_DIR, f"{target_date}.csv")
    odds_age_hours = (_time.time() - _os.path.getmtime(_csv_path)) / 3600 if _os.path.exists(_csv_path) else None

    predictions = []
    for match in matches:
        status = match.get("status", "")
        if status not in ("SCHEDULED", "TIMED", "IN_PLAY", "FINISHED", ""):
            continue
        home_id = match.get("homeTeam", {}).get("id")
        away_id = match.get("awayTeam", {}).get("id")
        if home_id is None or away_id is None:
            continue
        try:
            league_code = match.get("_league_code")
            home_name = match.get("homeTeam", {}).get("name", "")
            away_name = match.get("awayTeam", {}).get("name", "")
            match_odds = value_detector.get_match_odds(home_name, away_name, odds_map)
            referee_name = match.get("_referee")

            # Lineup impact — only when match kicks off within 3 hours
            lineup_impact = None
            try:
                from algorithms.lineup_impact import fetch_lineup, estimate_impact
                utc_str   = match.get("utcDate", "")
                kickoff   = datetime.strptime(utc_str, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
                hours_to_ko = (kickoff - datetime.now(timezone.utc)).total_seconds() / 3600
                if -1 < hours_to_ko < 3:   # window: 3h before to 1h after
                    match_id_val = match.get("id")
                    if match_id_val:
                        lu = fetch_lineup(match_id_val)
                        lineup_impact = estimate_impact(lu)
                        if lineup_impact and lineup_impact.get("notes"):
                            print(f"    [lineup] {home_name} vs {away_name}: {', '.join(lineup_impact['notes'])}")
            except Exception:
                pass

            pred = ensemble.predict_match(
                home_id, away_id, dc_params, elo_ratings, historical,
                reference_date=ref_date,
                league_code=league_code,
                standings_map=standings_map,
                market_odds=match_odds,
                elo_home_ratings=elo_home_ratings,
                elo_away_ratings=elo_away_ratings,
                odds_age_hours=odds_age_hours,
                referee_name=referee_name,
            )
            predictions.append({"match_info": match, "prediction": pred})
        except Exception as e:
            home = match.get("homeTeam", {}).get("name", "?")
            away = match.get("awayTeam", {}).get("name", "?")
            print(f"    [WARNING] {home} vs {away}: {e}")

    return ensemble.rank_predictions(predictions)


def print_day_summary(target_date: str, predictions: list[dict]):
    display = datetime.strptime(target_date, "%Y-%m-%d").strftime("%d/%m/%Y")
    print(f"\n  [{display}] {len(predictions)} partidos")
    for i, entry in enumerate(predictions[:3], 1):
        mi   = entry["match_info"]
        pred = entry["prediction"]
        home = mi.get("homeTeam", {}).get("name", "?")
        away = mi.get("awayTeam", {}).get("name", "?")
        outcome_map = {"home": f"{home} gana", "away": f"{away} gana", "draw": "Empate"}
        outcome = outcome_map.get(pred["best_outcome"], pred["best_outcome"])
        prob = pred["best_prob"] * 100
        stars = "★" * pred["stars"]
        print(f"    {i}. {home} vs {away} -> {outcome} {prob:.1f}% {stars}")


def main():
    args = parse_args()

    # Validate API key
    if API_KEY == "YOUR_API_KEY_HERE":
        print(
            "\n[ERROR] API key not configured.\n"
            "  1. Register for free at https://www.football-data.org/\n"
            "  2. Edit config.py and set API_KEY = 'your_key_here'\n"
        )
        sys.exit(1)

    # Determine target dates
    if args.date:
        target_dates = [datetime.strptime(args.date, "%Y-%m-%d").date()]
        mode = f"fecha concreta: {args.date}"
    else:
        target_dates = weekend_dates()
        labels = [d.strftime("%d/%m") for d in target_dates]
        mode = f"fin de semana: {' · '.join(labels)}"

    print(f"\n{'='*60}")
    print(f"  BetWinninGames  —  {mode}")
    print(f"{'='*60}\n")

    # Clear cache if requested
    if args.no_cache:
        import cache
        print("  Limpiando cache...")
        cache.clear_all()

    # --- Load historical data once (shared across all days) ---
    print("  [1/4] Cargando datos históricos (primera ejecución puede tardar)...")
    historical = fetcher.load_historical_data(args.league)
    print(f"        {len(historical)} partidos históricos cargados (football-data.org).")

    # --- Augment with football-data.co.uk older seasons (2020, 2021, 2022) ---
    print("  [1b2/4] Añadiendo temporadas históricas (football-data.co.uk)...")
    historical = fdco_fetcher.augment_historical(historical, args.league)
    print(f"        {len(historical)} partidos en total tras enriquecer.")

    if len(historical) < 50:
        print("  [WARNING] Pocos datos históricos. Las predicciones pueden ser menos fiables.")

    # --- Enrich historical with xG from Understat ---
    print("  [1c/4] Enriqueciendo con xG (Understat — solo primera ejecución por temporada)...")
    understat_fetcher.enrich_with_xg(historical)

    # --- Load current standings (for league table positions on cards) ---
    print("  [1b/4] Cargando clasificaciones...")
    leagues_to_load = {args.league: LEAGUES[args.league]} if args.league else LEAGUES
    standings_map: dict[int, int] = {}
    for code, comp_id in leagues_to_load.items():
        for entry in fetcher.get_standings(comp_id):
            tid = (entry.get("team") or {}).get("id")
            pos = entry.get("position")
            if tid and pos:
                standings_map[tid] = pos
    print(f"        {len(standings_map)} equipos con posición en tabla.")

    # --- Fit models once (shared across all days) ---
    print("  [2/4] Ajustando modelos...")
    ref_date = target_dates[-1]  # use last date in window as reference

    dc_params   = dixon_coles.fit_per_league(historical, reference_date=ref_date)
    elo_ratings = elo_module.build_ratings(historical)
    elo_home_rt, elo_away_rt = elo_module.build_split_ratings(historical)

    leagues_ok = sum(1 for k, v in dc_params.items() if k != "_global" and v)
    print(f"        Dixon-Coles: {leagues_ok} ligas ajustadas | "
          f"Elo: {len(elo_ratings)} equipos "
          f"(split: {len(elo_home_rt)} local / {len(elo_away_rt)} visitante)")

    # --- Fetch bookmaker odds for the full window (1 API call per league) ---
    print("  [3/4] Descargando cuotas (The Odds API)...")
    date_strs = [d.strftime("%Y-%m-%d") for d in target_dates]
    odds_fetcher.fetch_window(date_strs, args.league)

    # --- Generate predictions per day, collect into one JS ---
    print("  [4/4] Generando predicciones...")

    all_data: dict[str, dict] = {}
    for d in target_dates:
        date_str = d.strftime("%Y-%m-%d")
        predictions = predict_for_date(
            date_str, historical, dc_params, elo_ratings, args.league,
            standings_map=standings_map,
            elo_home_ratings=elo_home_rt,
            elo_away_ratings=elo_away_rt,
        )
        if not predictions:
            day_name = d.strftime("%A %d/%m")
            print(f"    {day_name}: sin partidos.")
            continue

        # --- AI Advisor: enrich picks with news-based adjustments ---
        if AI_ADVISOR_ENABLED:
            import ai_advisor
            predictions = ai_advisor.enrich_predictions(
                predictions, date_str, min_stars=AI_ADVISOR_MIN_STARS
            )

        odds_map   = value_detector.load_odds_csv(date_str)
        value_bets = value_detector.find_edges(predictions, odds_map)
        all_data[date_str] = {"predictions": predictions, "value_bets": value_bets}
        print_day_summary(date_str, predictions)

    # --- Write single JS for the visualiser ---
    print(f"\n{'='*60}")
    if all_data:
        output_path = generate_js(all_data, standings_map=standings_map)
        n_days    = len(all_data)
        n_matches = sum(len(v["predictions"]) for v in all_data.values())
        n_vbets   = sum(len(v["value_bets"])  for v in all_data.values())
        print(f"  {n_days} día(s) · {n_matches} partidos · {n_vbets} value bets")
        print(f"  -> {output_path}")

        # --- Persist picks to DB ---
        run_ts = datetime.now(timezone.utc).isoformat()
        total_saved = 0
        for date_str, day_data in all_data.items():
            saved = db_picks.save_picks(
                day_data["predictions"],
                date_str,
                run_ts,
                PICKS_DB,
                value_bets=day_data.get("value_bets"),
            )
            total_saved += saved
        if total_saved:
            print(f"  {total_saved} pick(s) nuevo(s) guardado(s) en {PICKS_DB}")

    else:
        print("  No se encontraron partidos para el período seleccionado.")
    print(f"{'='*60}\n")

    # --- Send Telegram notification ---
    if all_data:
        telegram_notifier.send_picks(all_data, quiet=False)

    # --- Run tracker (resolve results + generate tracker_data.js) ---
    tracker_module.run_tracker(quiet=True)


if __name__ == "__main__":
    main()
