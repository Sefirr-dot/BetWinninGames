"""
Match Context Classifier.

Assigns qualitative tags to a match based on Elo ratings, form, H2H history
and league table positions.  Tags are stored in picks_history.db and used by
tracker.py to compute per-tag ROI (which tags have systematic edge?).

Tags
----
even_match        Elo difference < 50 pts  → genuinely hard to call
top6_clash        Both teams in top 6 of the table
relegation_6ptr   Both teams in the bottom 6 (survival battle)
home_in_form      Home team form score > 0.70
away_in_form      Away team form score > 0.70
h2h_dominant      One team wins > 60% of H2H encounters (n >= 5)
"""

from __future__ import annotations

# Thresholds (easy to tune)
_ELO_EVEN_THRESHOLD   = 50     # max Elo diff for "even_match"
_FORM_STRONG          = 0.70   # form score threshold for "in_form" tag
_H2H_DOMINANCE        = 0.60   # min win rate for "h2h_dominant"
_H2H_MIN_N            = 5      # minimum H2H matches required
_TOP6_POSITIONS       = 6      # top-N positions count as "top6"
_RELEGATION_FROM_BOTTOM = 6    # bottom-N positions count as "relegation zone"


def classify(
    elo_pred: dict,
    form_pred: dict,
    h2h_pred: dict,
    home_pos: int | None,
    away_pos: int | None,
    league_size: int = 20,
) -> list[str]:
    """
    Return a list of context tags for a match.

    Parameters
    ----------
    elo_pred    : output of algorithms.elo.predict()
    form_pred   : output of algorithms.form.predict()
    h2h_pred    : output of algorithms.h2h.predict()
    home_pos    : league table position of home team (1=top), or None
    away_pos    : league table position of away team (1=top), or None
    league_size : number of teams in the league (default 20)
    """
    tags: list[str] = []

    # ── Elo proximity ───────────────────────────────────────────────────────
    rating_h = elo_pred.get("rating_home", 0)
    rating_a = elo_pred.get("rating_away", 0)
    if rating_h and rating_a and abs(rating_h - rating_a) < _ELO_EVEN_THRESHOLD:
        tags.append("even_match")

    # ── League table positions ───────────────────────────────────────────────
    if home_pos is not None and away_pos is not None:
        if home_pos <= _TOP6_POSITIONS and away_pos <= _TOP6_POSITIONS:
            tags.append("top6_clash")
        relegation_cutoff = league_size - _RELEGATION_FROM_BOTTOM + 1
        if home_pos >= relegation_cutoff and away_pos >= relegation_cutoff:
            tags.append("relegation_6ptr")

    # ── Form ────────────────────────────────────────────────────────────────
    home_form = form_pred.get("home_form", 0.0)
    away_form = form_pred.get("away_form", 0.0)
    if home_form >= _FORM_STRONG:
        tags.append("home_in_form")
    if away_form >= _FORM_STRONG:
        tags.append("away_in_form")

    # ── H2H dominance ───────────────────────────────────────────────────────
    if h2h_pred.get("sufficient"):
        n     = h2h_pred.get("n_matches", 0)
        hw    = h2h_pred.get("home_wins", 0)
        aw    = h2h_pred.get("away_wins", 0)
        if n >= _H2H_MIN_N:
            if hw / n >= _H2H_DOMINANCE or aw / n >= _H2H_DOMINANCE:
                tags.append("h2h_dominant")

    return tags
