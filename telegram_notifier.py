"""
Telegram notifications for BetWinninGames.

Sends up to 5 messages after each main.py run:
  1. Top picks (>= TELEGRAM_MIN_STARS) with outcome + cuota
  2. Value bets con outcome, edge y cuota de mercado
  3. Combinadas: Doble Favorita + Triple Equilibrada
  4. Combinadas: Cuadruple Valor + Quintuple Atrevida

Setup
-----
1. Create a bot via @BotFather -> copy the BOT_TOKEN to config.py
2. Start a chat with the bot (or add it to a channel/group)
3. Visit https://api.telegram.org/bot<TOKEN>/getUpdates -> copy the chat "id"
4. Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in config.py

The notifier is a no-op when either config value is empty.
"""

import math
import requests
from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, TELEGRAM_MIN_STARS

_OUTCOME_LABEL = {
    "home":   "Local",
    "draw":   "Empate",
    "away":   "Visitante",
    "over25": "Over 2.5",
    "btts":   "BTTS",
}
_STARS_EMOJI = {5: "🟢", 4: "🔵", 3: "🟡", 2: "🟠", 1: "⚪"}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _short(name: str, maxlen: int = 12) -> str:
    return name[:maxlen] if len(name) > maxlen else name


def _all_entries(all_data: dict) -> list[dict]:
    """Flatten all predictions across dates, sorted by stars DESC then prob DESC."""
    entries = []
    for date_str in sorted(all_data):
        for e in all_data[date_str].get("predictions", []):
            e["_date"] = date_str
            entries.append(e)
    entries.sort(key=lambda e: (
        -e["prediction"]["stars"],
        -e["prediction"]["best_prob"],
    ))
    return entries


def _all_vbs(all_data: dict) -> list[dict]:
    """Flatten all value bets sorted by edge DESC."""
    vbs = []
    for day in all_data.values():
        vbs.extend(day.get("value_bets", []))
    vbs.sort(key=lambda v: v.get("edge", 0), reverse=True)
    return vbs


# ---------------------------------------------------------------------------
# Parlay builder — mirrors calcBestBets() + getSuggestedParlays() in index.html
# ---------------------------------------------------------------------------

def _calc_best_bets(all_data: dict, vbs: list[dict] | None = None) -> list[dict]:
    """
    Build candidate bet pool from all predictions.

    Each bet is enriched with:
      - legScore = stars^2 * prob * (1 + edge_bonus)
      - edge_bonus from matching value bet (0 if none)
      - league, match_date for correlation penalty
    """
    # Build edge lookup from value bets: (home_name, away_name, outcome) -> edge
    _edge_map: dict[tuple, float] = {}
    if vbs:
        for vb in vbs:
            key = (vb.get("home_name", ""), vb.get("away_name", ""), vb.get("outcome", ""))
            prev = _edge_map.get(key, 0.0)
            if vb.get("edge", 0.0) > prev:
                _edge_map[key] = vb.get("edge", 0.0)

    bets = []
    for date_str in sorted(all_data):
        for e in all_data[date_str].get("predictions", []):
            pred  = e["prediction"]
            mi    = e["match_info"]
            stars = pred.get("stars", 1)

            p1 = pred.get("prob_home", 0) * 100
            px = pred.get("prob_draw", 0) * 100
            p2 = pred.get("prob_away", 0) * 100

            home_s  = _short(mi.get("homeTeam", {}).get("shortName") or mi.get("homeTeam", {}).get("name", "?"))
            away_s  = _short(mi.get("awayTeam", {}).get("shortName") or mi.get("awayTeam", {}).get("name", "?"))
            home_fn = mi.get("homeTeam", {}).get("name", "")
            away_fn = mi.get("awayTeam", {}).get("name", "")
            date_s  = mi.get("utcDate", "")[:10]
            match_key = date_s + "|" + home_fn
            league = mi.get("_league_code", "")

            # Victoria bet (home or away, highest 1X2 prob, skip draws)
            if p1 > px and p1 > p2 and p1 >= 48:
                edge_bonus = _edge_map.get((home_fn, away_fn, "home"), 0.0)
                leg_score = stars * stars * p1 * (1 + edge_bonus)
                bets.append({"entry": e, "type": "victoria", "match_key": match_key,
                    "label": f"Victoria {home_s}", "prob": p1,
                    "score": leg_score, "leg_score": leg_score,
                    "fair_odds": round(100 / p1, 2),
                    "edge": edge_bonus, "edge_bonus": edge_bonus,
                    "league": league, "match_date": date_s})
            elif p2 > p1 and p2 > px and p2 >= 48:
                edge_bonus = _edge_map.get((home_fn, away_fn, "away"), 0.0)
                leg_score = stars * stars * p2 * (1 + edge_bonus)
                bets.append({"entry": e, "type": "victoria", "match_key": match_key,
                    "label": f"Victoria {away_s}", "prob": p2,
                    "score": leg_score, "leg_score": leg_score,
                    "fair_odds": round(100 / p2, 2),
                    "edge": edge_bonus, "edge_bonus": edge_bonus,
                    "league": league, "match_date": date_s})

            # Over 2.5
            o25 = pred.get("over25", 0) * 100
            if pred.get("over25", 0) >= 0.48 and o25 >= 48:
                edge_bonus = _edge_map.get((home_fn, away_fn, "over25"), 0.0)
                leg_score = stars * stars * o25 * (1 + edge_bonus)
                bets.append({"entry": e, "type": "over25", "match_key": match_key,
                    "label": "Over 2.5 Goles", "prob": o25,
                    "score": leg_score, "leg_score": leg_score,
                    "fair_odds": round(100 / o25, 2),
                    "edge": edge_bonus, "edge_bonus": edge_bonus,
                    "league": league, "match_date": date_s})

            # BTTS
            btts = pred.get("btts_prob", 0) * 100
            if pred.get("btts_prob", 0) >= 0.48 and btts >= 48:
                edge_bonus = _edge_map.get((home_fn, away_fn, "btts"), 0.0)
                leg_score = stars * stars * btts * (1 + edge_bonus)
                bets.append({"entry": e, "type": "btts", "match_key": match_key,
                    "label": "Ambos Marcan", "prob": btts,
                    "score": leg_score, "leg_score": leg_score,
                    "fair_odds": round(100 / btts, 2),
                    "edge": edge_bonus, "edge_bonus": edge_bonus,
                    "league": league, "match_date": date_s})

    bets.sort(key=lambda b: b["leg_score"], reverse=True)
    return bets


def _corr_penalty(legs: list[dict]) -> float:
    """
    Correlation penalty para piernas del mismo día y liga.
    Empírico: rho ~0.03 -> factor 0.96 por par.
    """
    penalty = 1.0
    for i in range(len(legs)):
        for j in range(i + 1, len(legs)):
            a, b = legs[i], legs[j]
            if a.get("league") == b.get("league") and a.get("match_date") == b.get("match_date"):
                penalty *= 0.96
    return penalty


def _build_pool(bets: list[dict], min_stars: int, min_prob: float) -> list[dict]:
    """Unique-per-match pool filtered by min_stars and min_prob, sorted by prob desc."""
    seen, pool = set(), []
    for b in sorted(bets, key=lambda x: x["prob"], reverse=True):
        if b["entry"]["prediction"]["stars"] < min_stars:
            continue
        if b["prob"] < min_prob:
            continue
        if b["match_key"] not in seen:
            seen.add(b["match_key"])
            pool.append(b)
    return pool


def _exhaustive_search(pool: list[dict], n_legs: int, min_combined_prob: float) -> list[dict] | None:
    """
    Exhaustive search para maximizar prob_acumulada × corrPenalty.
    Integrar corrPenalty en la selección evita apilar piernas de la misma liga/día.
    Pool capped at 15 entries (C(15,5) = 3003 max combos).
    """
    from itertools import combinations

    if len(pool) < n_legs:
        return None

    search_pool = pool[:15]
    best_combo = None
    best_score = -1.0

    for combo in combinations(range(len(search_pool)), n_legs):
        legs = [search_pool[i] for i in combo]
        raw_prob = 1.0
        for leg in legs:
            raw_prob *= leg["prob"] / 100.0
        raw_prob *= 100.0
        if raw_prob < min_combined_prob:
            continue
        score = raw_prob * _corr_penalty(legs)  # penaliza correlación en la selección
        if score > best_score:
            best_score = score
            best_combo = legs

    return best_combo


def _parlay_text(bets: list[dict], title: str, emoji: str) -> list[str]:
    """Format one parlay using bet-specific label/prob/fair_odds."""
    combined_prob = 1.0
    combined_odds = 1.0
    has_odds      = True
    lines         = [f"{emoji} *{title}*"]

    for b in bets:
        pred    = b["entry"]["prediction"]
        mi      = b["entry"]["match_info"]
        home    = _short(mi.get("homeTeam", {}).get("shortName") or mi.get("homeTeam", {}).get("name", "?"))
        away    = _short(mi.get("awayTeam", {}).get("shortName") or mi.get("awayTeam", {}).get("name", "?"))
        stars_e = _STARS_EMOJI.get(pred["stars"], "")
        prob_f  = b["prob"] / 100
        fair    = b["fair_odds"]
        combined_prob *= prob_f
        if fair:
            combined_odds *= fair
        else:
            has_odds = False
        edge_str = f" +{b['edge_bonus']*100:.0f}%" if b.get("edge_bonus", 0) > 0.001 else ""
        lines.append(f"  {stars_e} `{home} vs {away}` — *{b['label']}* ({b['prob']:.0f}%{edge_str})")

    cp = _corr_penalty(bets)
    cp_str = f"{combined_prob*100:.1f}%"
    co_str = f"@{combined_odds:.2f}" if has_odds else ""
    corr_str = f" (corr: {cp:.2f})" if cp < 0.999 else ""
    lines.append(f"  _Prob combinada: {cp_str}  {co_str}{corr_str}_")
    return lines


def _build_parlays(all_data: dict, vbs: list[dict]) -> list[tuple]:
    """
    V2 Parlay Engine — mirrors getSuggestedParlays() in index.html.

    Uses exhaustive search over candidate pools to maximise:
      parlayEV = product(prob_i) * product(fairOdds_i) * correlationPenalty

    Each leg scored by: legScore = stars^2 * prob * (1 + edge_bonus)

    Returns list of (title, emoji, bets_list).
    """
    bets = _calc_best_bets(all_data, vbs=vbs)

    # Parlay configs: (n_legs, title, emoji, min_stars, min_prob, min_combined_prob)
    # Objetivo: maximizar probabilidad acumulada (seguras, aunque la cuota baje)
    configs = [
        (2, "Doble Segura",       "🟢", 3, 62, 38),
        (3, "Triple Sólida",      "🟡", 3, 58, 22),
        (4, "Cuádruple Firme",    "🟠", 2, 55, 12),
        (5, "Quíntuple Valiente", "🔴", 2, 55,  8),
    ]

    parlays = []
    for n_legs, title, emoji, min_stars, min_prob, min_cp in configs:
        pool = _build_pool(bets, min_stars, min_prob)
        best_legs = _exhaustive_search(pool, n_legs, min_cp)
        if best_legs:
            parlays.append((title, emoji, best_legs))

    return parlays


# ---------------------------------------------------------------------------
# Message builders
# ---------------------------------------------------------------------------

def _msg_picks(all_data: dict) -> str:
    """Message 1: top picks with outcome, prob and fair odds."""
    entries = _all_entries(all_data)
    top = [e for e in entries if e["prediction"]["stars"] >= TELEGRAM_MIN_STARS]
    if not top:
        return ""

    lines = ["🎯 *BetWinninGames — Picks del Fin de Semana*\n"]
    current_date = None

    for e in top:
        pred   = e["prediction"]
        mi     = e["match_info"]
        date_s = mi.get("utcDate", "")[:10]
        if date_s != current_date:
            current_date = date_s
            p = date_s.split("-")
            lines.append(f"\n📅 *{p[2]}/{p[1]}/{p[0]}*")

        home    = _short(mi.get("homeTeam", {}).get("shortName") or mi.get("homeTeam", {}).get("name", "?"))
        away    = _short(mi.get("awayTeam", {}).get("shortName") or mi.get("awayTeam", {}).get("name", "?"))
        league  = mi.get("_league_code", "")
        outcome = _OUTCOME_LABEL.get(pred["best_outcome"], pred["best_outcome"])
        prob    = pred["best_prob"]
        fair    = f"@{round(1/prob, 2)}" if prob > 0.01 else ""
        emoji   = _STARS_EMOJI.get(pred["stars"], "")

        lines.append(f"  {emoji} `{home} vs {away}` \\[{league}\\]")
        lines.append(f"      ▶ *{outcome}* — {prob*100:.1f}% {fair}")

    lines.append("\n_Solo informativo · Apuesta con responsabilidad_")
    return "\n".join(lines)


def _msg_valuebets(all_data: dict) -> str:
    """Message 2: value bets with outcome, edge and market odds."""
    vbs = _all_vbs(all_data)
    if not vbs:
        return ""

    lines = ["💎 *Value Bets — Edge vs mercado*\n"]
    for vb in vbs[:8]:
        home    = _short(vb.get("home_name", vb.get("match", "?")))
        away    = _short(vb.get("away_name", ""))
        league  = vb.get("league", "")
        outcome = _OUTCOME_LABEL.get(vb.get("outcome", ""), vb.get("outcome", ""))
        edge    = vb.get("edge", 0) * 100
        bk_odds = vb.get("bk_odds", 0)
        model_p = vb.get("model_prob", 0) * 100
        sharp   = " ⚡ Sharp" if vb.get("sharp_money") else ""
        match_str = f"{home} vs {away}" if away else home
        lines.append(
            f"  • `{match_str}` \\[{league}\\]{sharp}\n"
            f"    ▶ *{outcome}* — edge *+{edge:.1f}%* @{bk_odds:.2f} "
            f"_(modelo {model_p:.0f}%)_"
        )

    lines.append("\n_Solo informativo · Apuesta con responsabilidad_")
    return "\n".join(lines)


def _msg_winiela(all_data: dict) -> str:
    """Message: La Liga winiela — best pick per match (1/X/2/O25/BTTS)."""
    _pick_label  = {'1': 'Victoria Local', 'X': 'Empate', '2': 'Victoria Visit.',
                    'O25': 'Over 2.5', 'BTTS': 'Ambos marcan'}
    _pick_emoji  = {'1': '🟢', 'X': '🟡', '2': '🔵', 'O25': '🔵', 'BTTS': '🟠'}

    # Collect all PD matches sorted by date+time
    pd_entries = []
    for date_str in sorted(all_data):
        for e in all_data[date_str].get("predictions", []):
            if e["match_info"].get("_league_code") == "PD":
                pd_entries.append(e)
    pd_entries.sort(key=lambda e: e["match_info"].get("utcDate", ""))

    if len(pd_entries) < 2:
        return ""

    lines = ["⚽ *Quiniela La Liga — selección automática*\n"]
    combined_prob = 1.0
    combined_odds = 1.0

    for i, e in enumerate(pd_entries, 1):
        pred = e["prediction"]
        mi   = e["match_info"]
        home = _short(mi.get("homeTeam", {}).get("shortName") or mi.get("homeTeam", {}).get("name", "?"))
        away = _short(mi.get("awayTeam", {}).get("shortName") or mi.get("awayTeam", {}).get("name", "?"))
        t    = mi.get("utcDate", "")[11:16] or "?"

        p1  = pred.get("prob_home", 0) * 100
        px  = pred.get("prob_draw", 0) * 100
        p2  = pred.get("prob_away", 0) * 100
        po  = pred.get("over25",    0) * 100
        pb  = pred.get("btts_prob", 0) * 100

        candidates = [
            ("1",    p1,  pred.get("prob_home", 0)),
            ("X",    px,  pred.get("prob_draw", 0)),
            ("2",    p2,  pred.get("prob_away", 0)),
        ]
        if pred.get("over25", 0) >= 0.50:
            candidates.append(("O25",  po, pred.get("over25",    0)))
        if pred.get("btts_prob", 0) >= 0.50:
            candidates.append(("BTTS", pb, pred.get("btts_prob", 0)))

        best_pick, best_pct, best_prob = max(candidates, key=lambda c: c[1])
        fair = round(1.0 / best_prob, 2) if best_prob > 0.01 else None
        combined_prob *= best_prob
        if fair:
            combined_odds *= fair

        emoji = _pick_emoji.get(best_pick, "")
        label = _pick_label.get(best_pick, best_pick)
        lines.append(
            f"  *{i}.* `{home} vs {away}` {t}h\n"
            f"      {emoji} *{best_pick}* — {label} ({best_pct:.0f}%)"
            + (f" @{fair}" if fair else "")
        )

    cp = combined_prob * 100
    cp_str = f"{cp:.2f}%" if cp >= 0.1 else f"{cp:.4f}%"
    co_str = f"@{combined_odds:.1f}" if combined_odds < 100000 else f"@{combined_odds:.0e}"
    lines.append(f"\n_Prob combinada: {cp_str}  {co_str}_")
    lines.append("_Solo informativo · Apuesta con responsabilidad_")
    return "\n".join(lines)


def _msg_parlays(all_data: dict) -> list[str]:
    """Messages 3+: one message per parlay type."""
    vbs     = _all_vbs(all_data)
    parlays = _build_parlays(all_data, vbs)

    if not parlays:
        return []

    # Group into two messages: Doble+Triple in msg3, Cuadruple+Quintuple in msg4
    groups = [parlays[:2], parlays[2:]]
    messages = []

    for group in groups:
        if not group:
            continue
        lines = ["🎰 *Combinadas sugeridas*\n"]
        for title, emoji, bets in group:
            lines.extend(_parlay_text(bets, title, emoji))
            lines.append("")
        lines.append("_Solo informativo · Apuesta con responsabilidad_")
        messages.append("\n".join(lines))

    return messages


# ---------------------------------------------------------------------------
# Send helper
# ---------------------------------------------------------------------------

def _send(text: str, quiet: bool = False) -> bool:
    if not text.strip():
        return False
    try:
        resp = requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
            json={
                "chat_id":    TELEGRAM_CHAT_ID,
                "text":       text,
                "parse_mode": "Markdown",
            },
            timeout=15,
        )
        resp.raise_for_status()
        return True
    except Exception as exc:
        if not quiet:
            print(f"  [telegram] Warning: {exc}")
        return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def send_picks(all_data: dict, quiet: bool = False) -> bool:
    """
    Send up to 4 Telegram messages. Returns True if at least one was sent.
    Silently skips if token/chat are not configured.
    """
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return False

    sent = 0

    # Msg 1 — Top picks
    if _send(_msg_picks(all_data), quiet):
        sent += 1

    # Msg 2 — Value bets
    if _send(_msg_valuebets(all_data), quiet):
        sent += 1

    # Msg 3 & 4 — Parlays
    for text in _msg_parlays(all_data):
        if _send(text, quiet):
            sent += 1

    # Msg 5 — Winiela La Liga
    if _send(_msg_winiela(all_data), quiet):
        sent += 1

    if not quiet and sent:
        print(f"  [telegram] {sent} mensaje(s) enviado(s) a chat {TELEGRAM_CHAT_ID}.")

    return sent > 0
