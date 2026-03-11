"""
Pre-Match Sharpness Score — composite market intelligence signal.

Produces a score from 0 to 100 that quantifies how strongly the "smart money"
agrees with the model's prediction. A high Sharpness Score does NOT mean the
model is more accurate — it means there is convergent evidence from multiple
independent market signals that the edge is real.

Components (each contributes 0-25 points):
  1. CLV signal      — does our model prob beat the sharp closing line?
  2. Odds movement   — did the market move in our direction (steam)?
  3. Edge magnitude  — how large is the model edge vs market?
  4. MIS / staleness — is the market currently soft (fresh odds, no steam reversal)?

Research basis:
  - Pinnacle Solutions (2019): CLV > 0 is the best predictor of long-run profit
  - Kaunitz et al. (2017): value betting profitability correlated with market efficiency
  - The Professional Gambler Guide (Pinnacle): sharp money causes systematic odds movements

Usage:
    from algorithms.sharpness import compute_sharpness_score
    score = compute_sharpness_score(value_bet_dict, model_agreement_dict)

The score is attached to each value_bet dict as "sharpness_score" by value_detector.
Displayed in the visualizador as a badge on each value bet card.
"""

from __future__ import annotations
import math


def compute_sharpness_score(
    edge:            float,
    sharp_money:     bool,
    clv_vs_pinnacle: float | None,
    odds_movement:   float | None,
    mis:             float,
    dc_prob:         float | None = None,
    elo_prob:        float | None = None,
    model_prob:      float = 0.50,
    corr_discount:   float = 1.0,
) -> int:
    """
    Compute Sharpness Score (0-100) for a single value bet.

    Parameters
    ----------
    edge            : effective edge (model_prob - implied_prob), e.g. 0.12
    sharp_money     : True if odds shortened >10% on our outcome
    clv_vs_pinnacle : model_prob - pinnacle_implied_prob (positive = we beat Pinnacle)
    odds_movement   : opening / closing odds ratio (>1.0 = odds shortened = money in)
    mis             : Market Inefficiency Score [0.25, 1.50] from value_detector
    dc_prob         : Dixon-Coles probability for this outcome (for model agreement)
    elo_prob        : Elo probability for this outcome
    model_prob      : final ensemble blend probability
    corr_discount   : Kelly correlation discount [0.30, 1.0]; 1.0 = no correlated bets

    Returns
    -------
    Integer 0-100. Interpretation:
        0-25  : Little evidence of edge — bet with caution (low confidence)
        26-50 : Moderate signals — standard value bet
        51-75 : Strong signals — model + market aligned
        76-100: Exceptional convergence — highest-conviction bets
    """
    score = 0.0

    # ── Component 1: CLV vs Pinnacle (0-25 pts) ───────────────────────────
    # Beating the sharpest closing line is the gold standard of edge validation.
    # CLV of +5% = 25pts (full score). CLV < 0 = penalty.
    if clv_vs_pinnacle is not None:
        # Map [-0.10, +0.05] → [-25, +25], clamped to [0, 25]
        clv_pts = clv_vs_pinnacle / 0.05 * 25.0
        score += max(0.0, min(25.0, clv_pts))
    else:
        # No Pinnacle data: award partial credit based on edge alone
        score += max(0.0, min(12.0, edge / 0.12 * 12.0))

    # ── Component 2: Odds movement / steam (0-25 pts) ─────────────────────
    # Odds shortening in our direction = smart money on our side.
    if sharp_money:
        # Sharp money confirmed (>10% move): full bonus
        score += 20.0
        # Extra if very large movement
        if odds_movement and odds_movement >= 1.20:
            score += 5.0
    elif odds_movement is not None and odds_movement > 1.05:
        # Mild movement in our direction: partial credit, capped at 10pts
        movement_pts = min(10.0, (odds_movement - 1.05) / 0.05 * 10.0)
        score += movement_pts
    elif odds_movement is not None and odds_movement < 0.95:
        # Odds LENGTHENED on our outcome = public/sharp money against us
        score += 0.0   # no contribution (already penalised via MIS)

    score = min(score, 50.0)  # Components 1+2 capped at 50 so far

    # ── Component 3: Edge magnitude (0-25 pts) ────────────────────────────
    # Edge of 20%+ = full 25pts. Edge of 8% = 10pts.
    edge_pts = min(25.0, edge / 0.20 * 25.0)
    score += edge_pts

    # ── Component 4: Model agreement + MIS (0-25 pts) ─────────────────────
    # If DC and Elo both agree with the ensemble prediction → higher confidence.
    # Also uses MIS as a proxy for market softness.
    agreement_pts = 0.0

    # Sub-model agreement: DC and Elo both point same direction as final model
    if dc_prob is not None and elo_prob is not None:
        outcome_threshold = model_prob - 0.05   # both must be within 5pp of blend
        if dc_prob >= outcome_threshold and elo_prob >= outcome_threshold:
            agreement_pts += 10.0
        elif dc_prob >= outcome_threshold or elo_prob >= outcome_threshold:
            agreement_pts += 5.0
        # Penalise if sub-models strongly disagree (high dispersion)
        dispersion = abs(dc_prob - elo_prob)
        if dispersion > 0.15:
            agreement_pts -= 5.0

    # MIS contribution: [0.25, 1.50] → maps to [0, 10] pts
    mis_pts = min(10.0, (mis - 0.25) / 1.25 * 10.0)
    agreement_pts += mis_pts

    # Corr discount: if Kelly was discounted due to correlated bets, reduce confidence
    if corr_discount < 0.80:
        agreement_pts -= 3.0

    score += max(0.0, min(25.0, agreement_pts))

    # ── Final score ───────────────────────────────────────────────────────
    final = max(0, min(100, round(score)))
    return final


def sharpness_label(score: int) -> str:
    """Human-readable label for a sharpness score."""
    if score >= 80:
        return "Elite"
    elif score >= 65:
        return "Fuerte"
    elif score >= 50:
        return "Moderado"
    elif score >= 35:
        return "Bajo"
    else:
        return "Minimo"


def sharpness_color(score: int) -> str:
    """CSS variable name for the score color (used by frontend)."""
    if score >= 80:
        return "--accent-green"
    elif score >= 65:
        return "--bl1"     # teal
    elif score >= 50:
        return "--accent-yellow"
    elif score >= 35:
        return "--fl1"     # orange-ish
    else:
        return "--muted"
