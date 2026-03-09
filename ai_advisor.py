"""
AI Advisor — enriches picks with qualitative intelligence via Ollama.

For each pick with stars >= AI_ADVISOR_MIN_STARS:
  1. Fetches recent match news (injuries, suspensions, lineups) from Google News RSS.
  2. Sends match context + news to a local Ollama model (default: qwen2.5:9b).
  3. Parses the JSON response for probability adjustments + a human-readable note.
  4. Applies adjustments to prob_home/draw/away (capped at ±AI_ADVISOR_MAX_ADJ).
  5. Adds _ai_note and _ai_factors to the prediction dict for the visualizer.

Fails gracefully: if Ollama is down, the model is missing, or JSON parsing fails,
predictions are returned unchanged and a warning is printed.

Ollama must be running:  ollama serve
Your model must be pulled: ollama pull qwen2.5:9b
"""

import json
import time

import requests

import news_fetcher
from config import (
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    AI_ADVISOR_MAX_ADJ,
    AI_ADVISOR_TIMEOUT,
)

_CHAT_URL = f"{OLLAMA_BASE_URL}/api/chat"

_SYSTEM_PROMPT = (
    "You are a football betting analyst. Analyze news about upcoming matches and identify "
    "factors that could affect the statistical prediction: key player injuries, suspensions, "
    "lineup changes, fitness concerns, or other significant team news.\n"
    "Respond ONLY with valid JSON. No extra text, no markdown."
)

_USER_TEMPLATE = """/no_think
Match: {home} vs {away} ({league})
Date: {date}

Current statistical prediction:
  Home win: {ph:.1f}%  |  Draw: {pd:.1f}%  |  Away win: {pa:.1f}%
  Top pick: {outcome} at {prob:.1f}%  ({stars} stars)

Recent news headlines:
{news}

Your task:
- Identify the 1-3 most important factors from the news (injuries, suspensions, lineup changes).
- If a key home player is out → suggest negative adj_home (e.g. -0.05).
- If a key away player is out → suggest negative adj_away (e.g. -0.04).
- If no significant news, return 0.0 for both.
- Maximum adjustment: {max_adj:.0%} per side.
- Write the note in Spanish (one short sentence), or null if no relevant news.

Respond with this exact JSON (no other text):
{{
  "adj_home": <float between -{max_adj:.2f} and +{max_adj:.2f}>,
  "adj_away": <float between -{max_adj:.2f} and +{max_adj:.2f}>,
  "factors": [<up to 3 key factor strings in English>],
  "note": "<one sentence in Spanish or null>"
}}"""


def _call_ollama(prompt: str) -> dict | None:
    """
    POST to Ollama /api/chat with streaming + think:False and return parsed JSON.

    Uses streaming to avoid timeout issues with Qwen3's extended thinking mode.
    think:False disables the reasoning phase entirely for faster responses.
    """
    payload = {
        "model":  OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        "stream": True,
        "think":  False,      # disable Qwen3 extended thinking (much faster)
        "format": "json",     # force structured JSON output
        "options": {"temperature": 0.1},
    }
    try:
        tokens = []
        with requests.post(
            _CHAT_URL, json=payload,
            timeout=AI_ADVISOR_TIMEOUT, stream=True
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                chunk = json.loads(line)
                tok = chunk.get("message", {}).get("content", "")
                if tok:
                    tokens.append(tok)
                if chunk.get("done"):
                    break
        content = "".join(tokens)
        return json.loads(content)
    except requests.exceptions.ConnectionError:
        print("  [AI Advisor] Ollama no disponible (¿está corriendo? ollama serve)")
        return None
    except Exception as e:
        print(f"  [AI Advisor] Error llamando a Ollama: {e}")
        return None


def _apply_adjustment(pred: dict, adj_home: float, adj_away: float) -> None:
    """
    Apply probability adjustments in-place, re-normalising to sum = 1.0.

    adj_draw is inferred as -(adj_home + adj_away) so the total stays constant.
    All three are clamped so no outcome drops below 1%.
    """
    adj_home  = max(-AI_ADVISOR_MAX_ADJ, min(AI_ADVISOR_MAX_ADJ, adj_home))
    adj_away  = max(-AI_ADVISOR_MAX_ADJ, min(AI_ADVISOR_MAX_ADJ, adj_away))
    adj_draw  = -(adj_home + adj_away)

    new_ph = max(0.01, pred["prob_home"] + adj_home)
    new_pd = max(0.01, pred["prob_draw"] + adj_draw)
    new_pa = max(0.01, pred["prob_away"] + adj_away)

    total = new_ph + new_pd + new_pa
    pred["prob_home"] = new_ph / total
    pred["prob_draw"] = new_pd / total
    pred["prob_away"] = new_pa / total

    # Recompute best_outcome / best_prob after adjustment
    outcomes = [("home", pred["prob_home"]), ("draw", pred["prob_draw"]), ("away", pred["prob_away"])]
    best_label, best_p = max(outcomes, key=lambda x: x[1])
    pred["best_outcome"] = best_label
    pred["best_prob"]    = best_p


def _analyze_match(entry: dict, date_str: str) -> None:
    """
    Fetch news, call Ollama, and enrich the prediction in-place.
    Adds _ai_note and _ai_factors. May modify prob_home/draw/away.
    """
    mi   = entry["match_info"]
    pred = entry["prediction"]

    home = mi.get("homeTeam", {}).get("name", "Home")
    away = mi.get("awayTeam", {}).get("name", "Away")
    league = mi.get("_league_code", "")

    # 1 — Fetch news
    headlines = news_fetcher.fetch_match_news(home, away)
    if not headlines:
        news_text = "No recent news found."
    else:
        news_text = "\n".join(f"  - {h}" for h in headlines)

    # 2 — Build prompt
    prompt = _USER_TEMPLATE.format(
        home=home, away=away, league=league, date=date_str,
        ph=pred["prob_home"] * 100,
        pd=pred["prob_draw"] * 100,
        pa=pred["prob_away"] * 100,
        outcome=pred["best_outcome"],
        prob=pred["best_prob"] * 100,
        stars=pred["stars"],
        news=news_text,
        max_adj=AI_ADVISOR_MAX_ADJ,
    )

    # 3 — Call Ollama
    result = _call_ollama(prompt)
    if result is None:
        return

    # 4 — Validate and extract
    adj_home = float(result.get("adj_home", 0.0))
    adj_away = float(result.get("adj_away", 0.0))
    factors  = result.get("factors") or []
    note     = result.get("note") or None

    # Ignore noise: skip if total adjustment is tiny
    if abs(adj_home) + abs(adj_away) < 0.005:
        pred["_ai_note"]    = None
        pred["_ai_factors"] = []
        return

    # 5 — Apply and annotate
    old_ph = pred["prob_home"]
    old_pa = pred["prob_away"]
    _apply_adjustment(pred, adj_home, adj_away)

    pred["_ai_note"]    = note
    pred["_ai_factors"] = factors[:3]

    delta_h = (pred["prob_home"] - old_ph) * 100
    delta_a = (pred["prob_away"] - old_pa) * 100
    print(
        f"    [AI] {home[:15]} vs {away[:15]}  "
        f"Δhome={delta_h:+.1f}%  Δaway={delta_a:+.1f}%"
    )
    if note:
        print(f"         Nota: {note}")


def enrich_predictions(predictions: list[dict], date_str: str, min_stars: int = 3) -> list[dict]:
    """
    Enrich predictions with AI analysis for picks >= min_stars.

    Called from main.py after rank_predictions() and before generate_js().
    Returns the same list (modified in-place for qualifying picks).
    """
    qualifying = [e for e in predictions if e["prediction"].get("stars", 0) >= min_stars]

    if not qualifying:
        return predictions

    print(f"  [AI Advisor] Analizando {len(qualifying)} picks ({min_stars}+ estrellas)...")

    # Check Ollama is reachable before looping
    try:
        requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
    except Exception:
        print(f"  [AI Advisor] Ollama no responde en {OLLAMA_BASE_URL}. Saltando análisis.")
        return predictions

    t0 = time.time()
    for entry in qualifying:
        _analyze_match(entry, date_str)

    elapsed = time.time() - t0
    print(f"  [AI Advisor] Completado en {elapsed:.1f}s")

    return predictions
