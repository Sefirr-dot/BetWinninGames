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
# Parlay builder — mirrors calcBestBets() + getSuggestedParlays() in index.html
# ---------------------------------------------------------------------------

def _calc_best_bets(all_data: dict, vbs: list[dict] | None = None) -> list[dict]:
    """
    Mirror of calcBestBets() in index.html.
    Returns bets of type victoria/over25/btts sorted by score DESC.
    score = stars^2 * prob / 100  (same formula as the visualiser)

    When vbs is provided, each bet gets an "edge" field from the matching
    value bet so that _ilp_parlay() can use it in the ILP objective.
    """
    # Build edge lookup from value bets: (home_name, away_name, outcome) -> edge
    _edge_map: dict[tuple, float] = {}
    if vbs:
        for vb in vbs:
            key = (vb.get("home_name", ""), vb.get("away_name", ""), vb.get("outcome", ""))
            _edge_map[key] = vb.get("edge", 0.0)

    bets = []
    for date_str in sorted(all_data):
        for e in all_data[date_str].get("predictions", []):
            pred  = e["prediction"]
            mi    = e["match_info"]
            stars = pred.get("stars", 1)
            if stars < 2:
                continue

            p1 = pred.get("prob_home", 0) * 100
            px = pred.get("prob_draw", 0) * 100
            p2 = pred.get("prob_away", 0) * 100

            home_s  = _short(mi.get("homeTeam", {}).get("shortName") or mi.get("homeTeam", {}).get("name", "?"))
            away_s  = _short(mi.get("awayTeam", {}).get("shortName") or mi.get("awayTeam", {}).get("name", "?"))
            home_fn = mi.get("homeTeam", {}).get("name", "")
            away_fn = mi.get("awayTeam", {}).get("name", "")
            date_s  = mi.get("utcDate", "")[:10]
            match_key = date_s + "|" + home_fn

            # Victoria bet (home or away only, ≥55%)
            if p1 > px and p1 > p2 and p1 >= 55:
                bets.append({"entry": e, "type": "victoria", "match_key": match_key,
                    "label": f"Victoria {home_s}", "prob": p1,
                    "score": stars * stars * p1 / 100,
                    "fair_odds": round(100 / p1, 2),
                    "edge": _edge_map.get((home_fn, away_fn, "home"), 0.0)})
            elif p2 > p1 and p2 > px and p2 >= 55:
                bets.append({"entry": e, "type": "victoria", "match_key": match_key,
                    "label": f"Victoria {away_s}", "prob": p2,
                    "score": stars * stars * p2 / 100,
                    "fair_odds": round(100 / p2, 2),
                    "edge": _edge_map.get((home_fn, away_fn, "away"), 0.0)})

            # Over 2.5
            o25 = pred.get("over25", 0) * 100
            if pred.get("over25", 0) >= 0.50 and o25 >= 58:
                bets.append({"entry": e, "type": "over25", "match_key": match_key,
                    "label": "Over 2.5 Goles", "prob": o25,
                    "score": stars * stars * o25 / 100,
                    "fair_odds": round(100 / o25, 2),
                    "edge": _edge_map.get((home_fn, away_fn, "over25"), 0.0)})

            # BTTS
            btts = pred.get("btts_prob", 0) * 100
            if pred.get("btts_prob", 0) >= 0.50 and btts >= 58:
                bets.append({"entry": e, "type": "btts", "match_key": match_key,
                    "label": "Ambos Marcan", "prob": btts,
                    "score": stars * stars * btts / 100,
                    "fair_odds": round(100 / btts, 2),
                    "edge": _edge_map.get((home_fn, away_fn, "btts"), 0.0)})

    bets.sort(key=lambda b: b["score"], reverse=True)
    return bets


def _pool_from_bets(bets: list[dict], min_stars: int) -> list[dict]:
    """Unique-per-match pool sorted by prob DESC — mirrors buildPool() in JS."""
    seen, pool = set(), []
    for b in sorted(bets, key=lambda b: b["prob"], reverse=True):
        if b["entry"]["prediction"]["stars"] < min_stars:
            continue
        if b["match_key"] not in seen:
            seen.add(b["match_key"])
            pool.append(b)
    return pool


def _ilp_parlay(bets: list[dict], n_legs: int) -> list[dict]:
    """
    Select n_legs bets from the candidate pool using integer linear programming.

    Objective: maximise sum(log(prob_i/100) + 0.5 * edge_i)
    Constraints:
      - Exactly n_legs bets selected
      - At most 1 bet per match (match_key)
      - Binary selection variables

    Falls back to greedy selection if scipy.milp fails or is unavailable.
    Uses scipy.optimize.milp (scipy >= 1.7, already required by this project).
    """
    if len(bets) <= n_legs:
        return bets  # nothing to optimise

    try:
        import math
        import numpy as np
        from scipy.optimize import milp, LinearConstraint, Bounds

        n = len(bets)

        # Objective: minimise negative of maximise (milp minimises)
        c = np.array([
            -(math.log(max(b["prob"] / 100.0, 1e-9)) + 0.5 * b.get("edge", 0.0))
            for b in bets
        ])

        # Constraint 1: exactly n_legs selected
        A_total = np.ones((1, n))

        # Constraint 2: at most 1 per match
        unique_keys = list(dict.fromkeys(b["match_key"] for b in bets))
        A_match = np.zeros((len(unique_keys), n))
        for mi, mk in enumerate(unique_keys):
            for bi, b in enumerate(bets):
                if b["match_key"] == mk:
                    A_match[mi, bi] = 1.0

        A = np.vstack([A_total, A_match])
        lb = np.concatenate([[n_legs],   np.zeros(len(unique_keys))])
        ub = np.concatenate([[n_legs],   np.ones(len(unique_keys))])

        constraints = LinearConstraint(A, lb, ub)
        integrality = np.ones(n)
        bounds      = Bounds(lb=np.zeros(n), ub=np.ones(n))

        result = milp(c, constraints=constraints, integrality=integrality, bounds=bounds)
        if result.success:
            selected = [bets[i] for i in range(n) if result.x[i] > 0.5]
            if len(selected) == n_legs:
                return selected
    except Exception:
        pass

    # Greedy fallback: unique-per-match, sorted by score
    seen, pool = set(), []
    for b in sorted(bets, key=lambda b: b["score"], reverse=True):
        if b["match_key"] not in seen:
            seen.add(b["match_key"])
            pool.append(b)
        if len(pool) == n_legs:
            break
    return pool


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
        lines.append(f"  {stars_e} `{home} vs {away}` — *{b['label']}* ({b['prob']:.0f}%)")

    cp_str = f"{combined_prob*100:.1f}%"
    co_str = f"@{combined_odds:.2f}" if has_odds else ""
    lines.append(f"  _Prob combinada: {cp_str}  {co_str}_")
    return lines


def _build_parlays(all_data: dict, vbs: list[dict]) -> list[tuple]:
    """
    Mirror of getSuggestedParlays() in index.html.
    Returns list of (title, emoji, bets_list).

    Parlay selection now uses ILP (scipy.optimize.milp) to find the globally
    optimal N-leg combination maximising log(combined_prob) + 0.5*sum(edges),
    subject to exactly N bets and at most 1 per match.
    Falls back to greedy if ILP unavailable.
    """
    bets  = _calc_best_bets(all_data, vbs=vbs)
    pool4 = _pool_from_bets(bets, 4)
    pool3 = _pool_from_bets(bets, 3)
    pool2 = _pool_from_bets(bets, 2)

    parlays = []

    # Doble Segura — ILP from 4★+ pool (fall back to 3★+)
    dbl_src = pool4 if len(pool4) >= 2 else pool3
    if len(dbl_src) >= 2:
        parlays.append(("Doble Segura", "🟢", _ilp_parlay(dbl_src, 2)))

    # Triple Media — ILP from 3★+ pool
    if len(pool3) >= 3:
        parlays.append(("Triple Media", "🟡", _ilp_parlay(pool3, 3)))

    # Cuadruple Arriesgada — ILP from 3★+ (fall back to 2★+)
    quad_src = pool3 if len(pool3) >= 4 else pool2
    if len(quad_src) >= 4:
        parlays.append(("Cuadruple Arriesgada", "🟠", _ilp_parlay(quad_src, 4)))

    # EV+ — best VB per match, sorted by edge, ≥3★
    entry_idx = {}
    for b in bets:
        mk = b["match_key"]
        if mk not in entry_idx:
            entry_idx[mk] = b["entry"]

    ev_bets, ev_seen = [], set()
    for vb in sorted(vbs, key=lambda v: v.get("edge", 0), reverse=True):
        if len(ev_bets) >= 4:
            break
        mi   = None
        # find entry by home/away name
        home_n = vb.get("home_name", "")
        away_n = vb.get("away_name", "")
        for b in bets:
            bmi = b["entry"]["match_info"]
            if (bmi.get("homeTeam", {}).get("name", "") == home_n and
                    bmi.get("awayTeam", {}).get("name", "") == away_n):
                mi = b
                break
        if not mi or mi["match_key"] in ev_seen:
            continue
        if mi["entry"]["prediction"]["stars"] < 3:
            continue
        ev_seen.add(mi["match_key"])
        ev_bets.append({"entry": mi["entry"], "type": "victoria",
            "match_key": mi["match_key"],
            "label": _OUTCOME_LABEL.get(vb.get("outcome", ""), vb.get("outcome", "")),
            "prob": round(vb.get("model_prob", 0) * 100, 1),
            "score": 0,
            "fair_odds": round(vb["bk_odds"], 2) if vb.get("bk_odds") else None})

    if len(ev_bets) >= 2:
        parlays.append(("Valor EV+", "💰", ev_bets))

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

    # Group into two messages: Doble+Triple in msg3, Cuadruple+EV in msg4
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
