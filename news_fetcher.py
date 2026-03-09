"""
Match news fetcher for the AI Advisor.

Searches Google News RSS for recent news about each match:
injuries, suspensions, lineup changes, press conferences, etc.

No API key required — uses the public Google News RSS feed.
Fails silently so it never blocks the main prediction pipeline.
"""

import re
import xml.etree.ElementTree as ET
from urllib.parse import quote

import requests

# Prefixes / suffixes that bloat search queries
_STRIP = re.compile(
    r"\b(FC|CF|SC|AC|RC|SD|RCD|UD|CD|Athletic|Club|Fútbol|Futbol|United|City|Town|Rovers|Wanderers)\b",
    re.IGNORECASE,
)

_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; BetWinninGames/1.0)"}
_TIMEOUT = 10  # seconds


def _clean(name: str) -> str:
    """Strip common suffixes/prefixes that confuse news searches."""
    cleaned = _STRIP.sub("", name).strip()
    return cleaned if len(cleaned) >= 3 else name


def fetch_match_news(
    home_team: str,
    away_team: str,
    max_results: int = 6,
) -> list[str]:
    """
    Return up to max_results recent news headlines for a match.

    Searches Google News RSS with query:
        "{home} {away} injury lineup suspension team news"

    Returns a list of headline strings (may be empty on failure).
    """
    h = _clean(home_team)
    a = _clean(away_team)
    query = f"{h} {a} injury lineup suspension team news"
    url = (
        "https://news.google.com/rss/search"
        f"?q={quote(query)}&hl=en&gl=GB&ceid=GB:en"
    )

    try:
        resp = requests.get(url, timeout=_TIMEOUT, headers=_HEADERS)
        resp.raise_for_status()
        root = ET.fromstring(resp.content)

        snippets = []
        for item in root.findall(".//item")[:max_results]:
            title = (item.findtext("title") or "").strip()
            # Google News titles include source: "Headline - Source"
            # Strip the source part for cleaner output
            title = re.sub(r"\s+-\s+[^-]+$", "", title).strip()
            if title:
                snippets.append(title)

        return snippets

    except Exception:
        return []
