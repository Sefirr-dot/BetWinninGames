"""
Telegram notifications for BetWinninGames.

Sends top picks to a Telegram chat/channel after each main.py run.

Setup
-----
1. Create a bot via @BotFather → copy the BOT_TOKEN to config.py
2. Start a chat with the bot (or add it to a channel/group)
3. Visit https://api.telegram.org/bot<TOKEN>/getUpdates → copy the chat "id"
4. Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in config.py

The notifier is a no-op when either config value is empty.
"""

import requests
from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, TELEGRAM_MIN_STARS

_OUTCOME_LABEL = {
    "home": "Victoria Local",
    "draw": "Empate",
    "away": "Victoria Visitante",
}

_STARS_EMOJI = {5: "🟢", 4: "🔵", 3: "🟡", 2: "🟠", 1: "⚪"}


def _build_message(all_data: dict) -> str:
    """Build the Markdown message string from all_data (same format as main.py)."""
    lines = ["🎯 *BetWinninGames — Picks del Fin de Semana*\n"]

    total_picks = 0
    for date_str in sorted(all_data):
        day = all_data[date_str]
        top = [
            e for e in day.get("predictions", [])
            if e["prediction"]["stars"] >= TELEGRAM_MIN_STARS
        ]
        if not top:
            continue

        parts = date_str.split("-")
        lines.append(f"📅 *{parts[2]}/{parts[1]}/{parts[0]}*")

        for entry in top:
            mi   = entry["match_info"]
            pred = entry["prediction"]
            home   = mi.get("homeTeam", {}).get("shortName") or mi.get("homeTeam", {}).get("name", "?")
            away   = mi.get("awayTeam", {}).get("shortName") or mi.get("awayTeam", {}).get("name", "?")
            league = mi.get("_league_code", "")
            stars  = pred["stars"]
            outcome = _OUTCOME_LABEL.get(pred["best_outcome"], pred["best_outcome"])
            prob   = pred["best_prob"] * 100
            best_p = pred["best_prob"]
            fair   = f"@{round(1/best_p, 2)}" if best_p > 0.01 else ""
            emoji  = _STARS_EMOJI.get(stars, "")

            # Value bet indicator
            vb_line = ""
            for vb in (pred.get("dc") and [] or []):  # placeholder, real VBs below
                pass

            lines.append(
                f"  {emoji} `{home} vs {away}` \\[{league}\\]"
            )
            lines.append(
                f"      ▶ {outcome} — *{prob:.1f}%* {fair}"
            )
            total_picks += 1

        lines.append("")

    # Value bets summary
    all_vbs = []
    for date_str, day in all_data.items():
        all_vbs.extend(day.get("value_bets", []))
    if all_vbs:
        lines.append(f"💎 *{len(all_vbs)} value bet(s) detectado(s)*")
        for vb in all_vbs[:5]:  # show max 5
            lines.append(
                f"  • `{vb.get('match','')}` [{vb.get('league','')}] "
                f"edge={vb.get('edge',0)*100:.1f}% @{vb.get('bk_odds',0):.2f}"
            )
        lines.append("")

    if total_picks == 0:
        return ""

    lines.append("_Solo informativo · Apuesta con responsabilidad_")
    return "\n".join(lines)


def send_picks(all_data: dict, quiet: bool = False) -> bool:
    """
    Send picks to Telegram. Returns True if message was sent, False otherwise.
    Silently skips if token/chat are not configured.
    """
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return False

    text = _build_message(all_data)
    if not text:
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
        if not quiet:
            print(f"  [telegram] Picks enviados a chat {TELEGRAM_CHAT_ID}.")
        return True
    except Exception as exc:
        if not quiet:
            print(f"  [telegram] Warning: no se pudo enviar: {exc}")
        return False
