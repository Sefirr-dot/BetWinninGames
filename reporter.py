"""
TXT report generator.

Formats ensemble predictions into a human-readable file.
"""

import json
import os
from datetime import datetime
from config import OUTPUT_DIR, HIGH_CONFIDENCE_THRESHOLD, MEDIUM_CONFIDENCE_THRESHOLD, JS_OUTPUT_PATH

_ES_DAYS = {
    "Monday": "Lun", "Tuesday": "Mar", "Wednesday": "Mié",
    "Thursday": "Jue", "Friday": "Vie", "Saturday": "Sáb", "Sunday": "Dom",
}


def _stars(n: int) -> str:
    return "★" * n + "☆" * (5 - n)


def _outcome_label(outcome: str, home_name: str, away_name: str) -> str:
    if outcome == "home":
        return f"{home_name} gana"
    elif outcome == "away":
        return f"{away_name} gana"
    else:
        return "Empate"


def _pct(p: float) -> str:
    return f"{p * 100:.1f}%"


def _fair_odds(p: float) -> str:
    if p <= 0:
        return "N/A"
    return f"{1 / p:.2f}"


def _section(char: str, width: int = 60) -> str:
    return char * width


def _form_str(form_dict: dict) -> str:
    return form_dict.get("form_string", "?") if form_dict else "?"


def generate(
    predictions: list[dict],
    target_date: str,
    output_path: str | None = None,
    value_bets: list[dict] | None = None,
) -> str:
    """
    Write the predictions TXT and return the file path.

    Parameters
    ----------
    predictions : list of dicts, each containing:
        match_info  : original API match dict
        prediction  : output from ensemble.predict_match()
    target_date : str YYYY-MM-DD
    output_path : optional explicit output file path
    """
    if output_path is None:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        output_path = os.path.join(OUTPUT_DIR, f"predicciones_{target_date}.txt")

    display_date = datetime.strptime(target_date, "%Y-%m-%d").strftime("%d/%m/%Y")
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")

    lines = []

    def w(s=""):
        lines.append(s)

    # Header
    w(_section("═"))
    w(f"   BETWINNINGAMES - PREDICCIONES {display_date}")
    w("   Motor: Dixon-Coles + Elo + Forma + H2H + Fatiga (v2.1)")
    w(_section("═"))
    w()

    total = len(predictions)
    high_conf = [p for p in predictions if p["prediction"]["stars"] >= 4]
    med_conf  = [p for p in predictions if p["prediction"]["stars"] == 3]

    w(f"RESUMEN EJECUTIVO: {total} partidos analizados | "
      f"{len(high_conf)} picks de alta confianza")
    w()

    # --- HIGH CONFIDENCE ---
    if high_conf:
        five_star = [p for p in high_conf if p["prediction"]["stars"] == 5]
        four_star = [p for p in high_conf if p["prediction"]["stars"] == 4]
        star_label = _stars(5) if not four_star else f"{_stars(4)}–{_stars(5)}"
        w(_section("━"))
        w(f"  PICKS DE ALTA CONFIANZA ({star_label})")
        w(_section("━"))
        w()
        for rank, entry in enumerate(high_conf, 1):
            _write_match_block(w, rank, entry)

    # --- MEDIUM CONFIDENCE ---
    if med_conf:
        w(_section("━"))
        w(f"  PICKS DE CONFIANZA MEDIA ({_stars(3)})")
        w(_section("━"))
        w()
        for rank, entry in enumerate(med_conf, 1):
            _write_match_block(w, rank, entry)

    # --- VALUE BETS ---
    if value_bets:
        w(_section("━"))
        w(f"  VALUE BETS ({len(value_bets)} encontradas)")
        w(_section("━"))
        _write_value_bet_block(w, value_bets)

    # --- ALL MATCHES ---
    w(_section("━"))
    w("  TODOS LOS PARTIDOS")
    w(_section("━"))
    w()
    _write_summary_table(w, predictions)

    # Footer
    w()
    w(_section("═"))
    w(f"Generado: {now_str} | Datos: football-data.org")
    w("AVISO: Solo con fines informativos/educativos.")
    w(_section("═"))

    text = "\n".join(lines)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)

    return output_path


def _write_match_block(w, rank: int, entry: dict) -> None:
    mi   = entry["match_info"]
    pred = entry["prediction"]

    home_name  = mi.get("homeTeam", {}).get("name", "Home")
    away_name  = mi.get("awayTeam", {}).get("name", "Away")
    league     = mi.get("_league_code", "")
    match_time = mi.get("utcDate", "")
    try:
        match_time = datetime.strptime(match_time, "%Y-%m-%dT%H:%M:%SZ").strftime("%H:%M")
    except Exception:
        match_time = "TBD"

    outcome_label = _outcome_label(pred["best_outcome"], home_name, away_name)

    # Corners line
    corn = pred.get("corners", {})
    exp_corners = corn.get("expected_corners", 0)
    over95_prob = corn.get("over_lines", {}).get(9.5, 0)
    over_corners_label = "Over 9.5" if over95_prob >= 0.50 else "Under 9.5"

    # Form strings
    h_form = _form_str(pred.get("form", {}).get("home_form"))
    a_form = _form_str(pred.get("form", {}).get("away_form"))

    elo_conf = pred.get("elo", {})
    elo_home_pct = _pct(elo_conf.get("prob_home", 0)) if elo_conf else "N/A"

    w(f"[{rank}] {home_name} vs {away_name} | {league} | {match_time}")
    w(f"    RESULTADO:   {outcome_label:<20} → {_pct(pred['best_prob'])} | "
      f"Cuota justa: {_fair_odds(pred['best_prob'])}")
    w(f"    OVER 2.5:    {'SI' if pred['over25'] >= 0.50 else 'NO':<20}"
      f"→ {_pct(pred['over25'])}")
    w(f"    BTTS:        {'Ambos marcan' if pred['btts_prob'] >= 0.50 else 'No ambos':<20}"
      f"→ {_pct(pred['btts_prob'])}")
    w(f"    CORNERS:     {over_corners_label:<20} → {exp_corners:.1f} esperados")
    w(f"    Goles esp:   {home_name} {pred['expected_goals_home']:.2f} "
      f"- {away_name} {pred['expected_goals_away']:.2f}")
    w(f"    Confianza Elo: {elo_home_pct} local | Forma: {home_name} {h_form} | {away_name} {a_form}")
    w(f"    Score máx probable: {pred['most_likely_score'][0]}-{pred['most_likely_score'][1]} "
      f"({_pct(pred['most_likely_score_prob'])})")

    # H2H record (only when sufficient historical data)
    h2h_data = pred.get("h2h", {})
    if h2h_data.get("sufficient"):
        n   = h2h_data["n_matches"]
        hw  = h2h_data["home_wins"]
        d   = h2h_data["draws"]
        aw  = h2h_data["away_wins"]
        dom = (f"{home_name} domina" if hw > aw + 1
               else f"{away_name} domina" if aw > hw + 1
               else "Muy igualado")
        w(f"    H2H ({n} ptds):   {home_name} {hw}V {d}E {aw}D  → {dom}")

        # Contradiction alert: H2H favours the opposite team to the ensemble pick
        ensemble_winner = pred["best_outcome"]
        h2h_probs = {
            "home": h2h_data.get("prob_home", 0),
            "draw": h2h_data.get("prob_draw", 0),
            "away": h2h_data.get("prob_away", 0),
        }
        h2h_winner = max(h2h_probs, key=h2h_probs.get)
        if (ensemble_winner in ("home", "away")
                and h2h_winner in ("home", "away")
                and ensemble_winner != h2h_winner
                and pred["best_prob"] >= 0.55):
            h2h_fav = home_name if h2h_winner == "home" else away_name
            w(f"    ! ATENCION: H2H contradice la prediccion (historico favorece a {h2h_fav})")

    # AI Advisor note (shown when Ollama found relevant news)
    ai_note    = pred.get("_ai_note")
    ai_factors = pred.get("_ai_factors") or []
    if ai_note or ai_factors:
        w(f"    * IA:         {ai_note or ''}")
        if ai_factors:
            w(f"      Factores: {' | '.join(ai_factors)}")

    # Fatigue warning (shown when either team has < 5 days rest)
    fat = pred.get("fatigue", {})
    h_days = fat.get("home_days")
    a_days = fat.get("away_days")
    warnings = []
    if h_days is not None and h_days < 5:
        warnings.append(f"{home_name} {h_days}d descanso")
    if a_days is not None and a_days < 5:
        warnings.append(f"{away_name} {a_days}d descanso")
    if warnings:
        w(f"    ! Fatiga:     {' | '.join(warnings)}")

    w("    " + "-" * 56)
    w()


def _write_value_bet_block(w, value_bets: list[dict]) -> None:
    """Display value bets with edge, odds and Kelly fraction."""
    w()
    for i, vb in enumerate(value_bets, 1):
        outcome_label = {
            "home": f"{vb['home_name']} gana",
            "away": f"{vb['away_name']} gana",
            "draw": "Empate",
        }.get(vb["outcome"], vb["outcome"])
        w(f"  [{i}] {vb['match']}  ({vb['league']})")
        w(f"      Apuesta:   {outcome_label}")
        w(f"      Edge:      {vb['edge'] * 100:.1f}%  |  "
          f"Cuota: {vb['bk_odds']:.2f}  |  "
          f"Kelly: {vb['kelly_fraction'] * 100:.1f}%")
        w(f"      Modelo:    {_pct(vb['model_prob'])}  vs  "
          f"Implícita: {_pct(vb['implied_prob'])}")
        w()


def _write_summary_table(w, predictions: list[dict]) -> None:
    header = f"{'#':<3} {'Partido':<35} {'Liga':<5} {'1':<7} {'X':<7} {'2':<7} {'O2.5':<7} {'BTTS':<7} {'★'}"
    w(header)
    w("-" * len(header))
    for i, entry in enumerate(predictions, 1):
        mi   = entry["match_info"]
        pred = entry["prediction"]
        home = mi.get("homeTeam", {}).get("shortName") or mi.get("homeTeam", {}).get("name", "?")[:12]
        away = mi.get("awayTeam", {}).get("shortName") or mi.get("awayTeam", {}).get("name", "?")[:12]
        match_str = f"{home} vs {away}"
        league    = mi.get("_league_code", "")

        w(
            f"{i:<3} {match_str:<35} {league:<5} "
            f"{_pct(pred['prob_home']):<7} "
            f"{_pct(pred['prob_draw']):<7} "
            f"{_pct(pred['prob_away']):<7} "
            f"{_pct(pred['over25']):<7} "
            f"{_pct(pred['btts_prob']):<7} "
            f"{_stars(pred['stars'])}"
        )
    w()


# ---------------------------------------------------------------------------
# JS output (v3.0 — feeds visualizador/index.html)
# ---------------------------------------------------------------------------

def _make_short(name: str) -> str:
    """First word, or full name if it fits in 12 chars."""
    return name if len(name) <= 12 else name.split()[0]


def _prediction_to_dict(entry: dict, date_str: str, match_vbs: list[dict], standings_map: dict | None = None) -> dict:
    """Serialise one ensemble entry to the ALL_MATCHES JS object format."""
    mi   = entry["match_info"]
    pred = entry["prediction"]

    home_name  = mi.get("homeTeam", {}).get("name", "")
    away_name  = mi.get("awayTeam", {}).get("name", "")
    home_short = mi.get("homeTeam", {}).get("shortName") or _make_short(home_name)
    away_short = mi.get("awayTeam", {}).get("shortName") or _make_short(away_name)

    # Match kick-off time (UTC → HH:MM; None when not available)
    match_time = None
    try:
        match_time = datetime.strptime(
            mi.get("utcDate", ""), "%Y-%m-%dT%H:%M:%SZ"
        ).strftime("%H:%M")
    except Exception:
        pass

    # Corners
    corn       = pred.get("corners") or {}
    exp_corn   = corn.get("expected_corners")
    over95     = corn.get("over_lines", {}).get(9.5, 0)

    # Cards
    cards_data  = pred.get("cards") or {}
    exp_cards   = cards_data.get("expected_cards")
    home_cards  = cards_data.get("home_cards")
    away_cards  = cards_data.get("away_cards")
    cards_over35 = cards_data.get("over_lines", {}).get(3.5, 0)

    # Form strings
    form_data  = pred.get("form") or {}
    h_form     = (form_data.get("home_form") or {}).get("form_string")
    a_form     = (form_data.get("away_form") or {}).get("form_string")

    # Elo home-win probability
    elo_data   = pred.get("elo") or {}
    elo_conf   = round(elo_data.get("prob_home", 0) * 100, 1) if elo_data else None

    # Most-likely score
    mls        = pred.get("most_likely_score")
    best_score = f"{mls[0]}-{mls[1]}" if mls else None
    best_score_prob = round((pred.get("most_likely_score_prob") or 0) * 100, 1)

    # Fair odds for the best outcome
    best_prob  = pred.get("best_prob", 0)
    fair_odds  = round(1 / best_prob, 2) if best_prob > 0.01 else None

    # Market odds for the best outcome (from value bets when available)
    best_outcome = pred.get("best_outcome")
    market_odds  = None
    for vb in match_vbs:
        if vb["outcome"] == best_outcome:
            market_odds = round(vb["bk_odds"], 2)
            break

    # H2H summary
    h2h_data = pred.get("h2h") or {}
    h2h_str  = None
    if h2h_data.get("sufficient"):
        hw, dr, aw = h2h_data["home_wins"], h2h_data["draws"], h2h_data["away_wins"]
        if hw > aw + 1:
            dom = f"{home_short} domina"
        elif aw > hw + 1:
            dom = f"{away_short} domina"
        else:
            dom = "Igualado"
        h2h_str = f"{home_short} {hw}V {dr}E {aw}D → {dom}"

    # Sub-model probabilities (for modal breakdown in the visualiser)
    def _sub_probs(d: dict) -> dict | None:
        if not d:
            return None
        ph = d.get("prob_home", 0)
        pd_ = d.get("prob_draw", 0)
        pa  = d.get("prob_away", 0)
        if ph == pd_ == pa == 0:
            return None
        return {"p1": round(ph * 100, 1), "pX": round(pd_ * 100, 1), "p2": round(pa * 100, 1)}

    sub_models: dict = {}
    dc_sub = _sub_probs(pred.get("dc"))
    if dc_sub:
        sub_models["dc"] = dc_sub
    elo_sub = _sub_probs(pred.get("elo"))
    if elo_sub:
        sub_models["elo"] = elo_sub
    form_sub = _sub_probs(pred.get("form"))
    if form_sub:
        sub_models["form"] = form_sub
    h2h_sub = _sub_probs(h2h_data) if h2h_data.get("sufficient") else None
    if h2h_sub:
        sub_models["h2h"] = h2h_sub

    # Elo ratings for both teams
    elo_data_full = pred.get("elo") or {}
    elo_home_rating = round(elo_data_full.get("rating_home", 0)) if elo_data_full else None
    elo_away_rating = round(elo_data_full.get("rating_away", 0)) if elo_data_full else None

    # Fatigue (days since last match)
    fat_data = pred.get("fatigue") or {}
    fatigue_home = fat_data.get("home_days")
    fatigue_away = fat_data.get("away_days")

    # League table positions
    home_id = mi.get("homeTeam", {}).get("id")
    away_id = mi.get("awayTeam", {}).get("id")
    home_pos = standings_map.get(home_id) if standings_map and home_id else None
    away_pos = standings_map.get(away_id) if standings_map and away_id else None

    # Score grid from Dixon-Coles prob matrix (6×6)
    dc_pred = pred.get("dc") or {}
    matrix = dc_pred.get("prob_matrix")
    score_grid = None
    if matrix is not None:
        try:
            score_grid = [
                [round(float(matrix[h][a]) * 100, 2) for a in range(min(6, matrix.shape[1]))]
                for h in range(min(6, matrix.shape[0]))
            ]
        except Exception:
            pass

    # Value bets for this specific match
    vb_list = [
        {
            "outcome":     vb["outcome"],
            "edge":        round(vb["edge"] * 100, 1),
            "bkOdds":      round(vb["bk_odds"], 2),
            "kelly":       round(vb["kelly_fraction"] * 100, 1),
            "modelProb":   round(vb["model_prob"] * 100, 1),
            "impliedProb": round(vb["implied_prob"] * 100, 1),
        }
        for vb in match_vbs
    ] or None

    return {
        "date":          date_str,
        "home":          home_name,
        "homeShort":     home_short,
        "away":          away_name,
        "awayShort":     away_short,
        "league":        mi.get("_league_code", ""),
        "time":          match_time,
        "stars":         pred.get("stars", 1),
        "prob1":         round(pred.get("prob_home", 0) * 100, 1),
        "probX":         round(pred.get("prob_draw", 0) * 100, 1),
        "prob2":         round(pred.get("prob_away", 0) * 100, 1),
        "over25":        round(pred.get("over25", 0) * 100, 1),
        "over25yn":      pred.get("over25", 0) >= 0.50,
        "btts":          round(pred.get("btts_prob", 0) * 100, 1),
        "bttsyn":        pred.get("btts_prob", 0) >= 0.50,
        "corners":       round(exp_corn, 1) if exp_corn is not None else None,
        "cornersOver":   bool(over95 >= 0.50) if exp_corn is not None else None,
        "cards":         round(exp_cards, 1) if exp_cards is not None else None,
        "cardsHome":     round(home_cards, 1) if home_cards is not None else None,
        "cardsAway":     round(away_cards, 1) if away_cards is not None else None,
        "cardsOver":     bool(cards_over35 >= 0.50) if exp_cards is not None else None,
        "goalsHome":     round(pred.get("expected_goals_home", 0), 2),
        "goalsAway":     round(pred.get("expected_goals_away", 0), 2),
        "eloConf":       elo_conf,
        "formHome":      h_form,
        "formAway":      a_form,
        "bestScore":     best_score,
        "bestScoreProb": best_score_prob,
        "fairOdds":      fair_odds,
        "marketOdds":    market_odds,
        "h2h":           h2h_str,
        "valueBets":     vb_list,
        "subModels":     sub_models or None,
        "eloHome":       elo_home_rating,
        "eloAway":       elo_away_rating,
        "fatigueHome":   fatigue_home,
        "fatigueAway":   fatigue_away,
        "homePos":       home_pos,
        "awayPos":       away_pos,
        "scoreGrid":     score_grid,
        "aiNote":        pred.get("_ai_note"),
        "aiFactors":     pred.get("_ai_factors") or None,
    }


def generate_js(
    all_data: dict,
    output_path: str | None = None,
    standings_map: dict | None = None,
) -> str:
    """
    Write visualizador/data/predictions.js from all predictions in the window.

    Parameters
    ----------
    all_data    : {date_str: {"predictions": list[dict], "value_bets": list[dict]}}
    output_path : override the default JS_OUTPUT_PATH from config

    Returns the path written.
    """
    if output_path is None:
        output_path = JS_OUTPUT_PATH

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    dates = sorted(all_data.keys())

    # Build DATE_LABELS with Spanish day names
    date_labels = {}
    for d_str in dates:
        d = datetime.strptime(d_str, "%Y-%m-%d").date()
        day_es = _ES_DAYS.get(d.strftime("%A"), d.strftime("%a"))
        date_labels[d_str] = f"{day_es} {d.strftime('%d/%m')}"

    # Build ALL_MATCHES — index value bets by (home_name, away_name) per day
    all_matches = []
    for date_str in dates:
        day   = all_data[date_str]
        preds = day.get("predictions", [])
        vbs   = day.get("value_bets", [])

        vb_index: dict[tuple, list] = {}
        for vb in vbs:
            key = (vb.get("home_name", ""), vb.get("away_name", ""))
            vb_index.setdefault(key, []).append(vb)

        for entry in preds:
            mi        = entry["match_info"]
            home_name = mi.get("homeTeam", {}).get("name", "")
            away_name = mi.get("awayTeam", {}).get("name", "")
            match_vbs = vb_index.get((home_name, away_name), [])
            all_matches.append(_prediction_to_dict(entry, date_str, match_vbs, standings_map))

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    js = (
        f"// Auto-generated by BetWinninGames — {now_str}\n"
        f"// Run `python main.py` to refresh.\n"
        f"const ALL_MATCHES = {json.dumps(all_matches, ensure_ascii=False, indent=2)};\n\n"
        f"const DATES = {json.dumps(dates, ensure_ascii=False)};\n\n"
        f"const DATE_LABELS = {json.dumps(date_labels, ensure_ascii=False)};\n"
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(js)

    return output_path
