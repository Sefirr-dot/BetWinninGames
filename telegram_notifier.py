"""
Telegram notifications for BetWinninGames.

Sends 4 messages after each main.py run:
  1. Top picks (>= TELEGRAM_MIN_STARS) with outcome + cuota
  2. Value bets con outcome, edge y cuota de mercado
  3. Combinadas: Doble Segura + Triple Media
  4. Combinadas: Cuadruple Arriesgada + Valor EV+

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
# Parlay builder (mirrors getSuggestedParlays in index.html)
# ---------------------------------------------------------------------------

def _build_pool(entries: list[dict], min_stars: int) -> list[dict]:
    """Unique-per-match pool of bets, sorted by prob DESC."""
    seen, pool = set(), []
    for e in entries:
        pred = e["prediction"]
        if pred["stars"] < min_stars:
            continue
        mi  = e["match_info"]
        key = mi.get("utcDate", "")[:10] + "|" + mi.get("homeTeam", {}).get("name", "")
        if key not in seen:
            seen.add(key)
            pool.append(e)
    return pool


def _parlay_text(legs: list[dict], title: str, emoji: str) -> list[str]:
    """Format one parlay as a list of message lines."""
    combined_prob  = 1.0
    combined_odds  = 1.0
    has_odds       = True
    lines          = [f"{emoji} *{title}*"]

    for e in legs:
        pred    = e["prediction"]
        mi      = e["match_info"]
        home    = _short(mi.get("homeTeam", {}).get("shortName") or mi.get("homeTeam", {}).get("name", "?"))
        away    = _short(mi.get("awayTeam", {}).get("shortName") or mi.get("awayTeam", {}).get("name", "?"))
        outcome = _OUTCOME_LABEL.get(pred["best_outcome"], pred["best_outcome"])
        prob    = pred["best_prob"]
        fair    = round(1.0 / prob, 2) if prob > 0.01 else None
        stars_e = _STARS_EMOJI.get(pred["stars"], "")
        combined_prob *= prob
        if fair:
            combined_odds *= fair
        else:
            has_odds = False
        lines.append(f"  {stars_e} `{home} vs {away}` — *{outcome}* ({prob*100:.0f}%)")

    cp_str = f"{combined_prob*100:.1f}%"
    co_str = f"@{combined_odds:.2f}" if has_odds else ""
    lines.append(f"  _Prob combinada: {cp_str}  {co_str}_")
    return lines


def _build_parlays(entries: list[dict], vbs: list[dict]) -> list[tuple[str, str, list[dict]]]:
    """
    Returns list of (title, emoji, legs_entries) for:
      - Doble Segura   (2 legs, 4★+ pref)
      - Triple Media   (3 legs, 3★+)
      - Cuadruple Arriesgada (4 legs, 3★+ / fallback 2★)
      - Valor EV+      (VB picks, 2-4 legs)
    """
    pool4 = _build_pool(entries, 4)
    pool3 = _build_pool(entries, 3)
    pool2 = _build_pool(entries, 2)

    parlays = []

    # Doble
    dbl = pool4 if len(pool4) >= 2 else pool3
    if len(dbl) >= 2:
        parlays.append(("Doble Segura", "🟢", dbl[:2]))

    # Triple
    if len(pool3) >= 3:
        parlays.append(("Triple Media", "🟡", pool3[:3]))

    # Cuadruple
    quad = pool3 if len(pool3) >= 4 else pool2
    if len(quad) >= 4:
        parlays.append(("Cuadruple Arriesgada", "🟠", quad[:4]))

    # EV+ (reuse match_info from entries where VBs exist)
    # Build index: (home_name, away_name) -> entry
    entry_index = {}
    for e in entries:
        mi = e["match_info"]
        key = (mi.get("homeTeam", {}).get("name", ""), mi.get("awayTeam", {}).get("name", ""))
        entry_index[key] = e

    ev_legs, ev_seen = [], set()
    for vb in sorted(vbs, key=lambda v: v.get("edge", 0), reverse=True):
        if len(ev_legs) >= 4:
            break
        key = (vb.get("home_name", ""), vb.get("away_name", ""))
        if key in ev_seen:
            continue
        e = entry_index.get(key)
        if e and e["prediction"]["stars"] >= 3:
            ev_seen.add(key)
            ev_legs.append((e, vb))

    if len(ev_legs) >= 2:
        parlays.append(("Valor EV+", "💰", [leg[0] for leg in ev_legs]))

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
    entries = _all_entries(all_data)
    vbs     = _all_vbs(all_data)
    parlays = _build_parlays(entries, vbs)

    if not parlays:
        return []

    # Group into two messages: Doble+Triple in msg3, Cuadruple+EV in msg4
    groups = [parlays[:2], parlays[2:]]
    messages = []

    for group in groups:
        if not group:
            continue
        lines = ["🎰 *Combinadas sugeridas*\n"]
        for title, emoji, legs in group:
            lines.extend(_parlay_text(legs, title, emoji))
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
