"""
Weather impact model for expected goals (lambda/mu).

Academic basis:
  - Healy & Pate (2011): heavy rain reduces total goals by ~8-12%
  - Strudwick (2016) sports science: wind > 40 km/h degrades pass accuracy
    and reduces attacking effectiveness by ~7-10%
  - Temperature < 2°C: muscle performance drops ~3-5% (cold-start effect)
  - Combined adverse conditions: effects are additive but capped

Inputs are the raw weather dict from weather_fetcher.fetch_weather().
Output: {"lam_mult": float, "mu_mult": float, "draw_boost": float, "notes": list[str]}

The draw_boost is a probability addend (not multiplier) applied to prob_draw.
When expected goals drop, both teams score less → empates más probables.
This is handled in ensemble.py BEFORE the draw model kick in.

Design philosophy:
  - Conservative adjustments (max ±12%) to avoid overriding the core model
  - Only activates for genuinely extreme conditions (top ~10-15% of matches)
  - All multipliers ≥ 0.88 and ≤ 1.12 (symmetric, never too dramatic)
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Thresholds (calibrated to match academic literature)
# ---------------------------------------------------------------------------

_WIND_MODERATE     = 25.0   # km/h — noticeable impact on long balls
_WIND_STRONG       = 40.0   # km/h — significant passing disruption
_WIND_VERY_STRONG  = 55.0   # km/h — extreme conditions

_RAIN_LIGHT        = 0.5    # mm/h — some impact
_RAIN_HEAVY        = 2.0    # mm/h — significant impact
_RAIN_VERY_HEAVY   = 5.0    # mm/h — extreme conditions

_TEMP_COLD         = 5.0    # °C — cold conditions
_TEMP_VERY_COLD    = 0.0    # °C — freezing, grass hard
_TEMP_HOT          = 28.0   # °C — heat impact on high-intensity press
_TEMP_VERY_HOT     = 33.0   # °C — extreme heat


def compute_weather_impact(weather: dict | None) -> dict:
    """
    Convert weather conditions to lambda/mu multipliers.

    Parameters
    ----------
    weather : from weather_fetcher.fetch_weather(), or None

    Returns
    -------
    dict:
        lam_mult   : float in [0.88, 1.12] — multiplier for home expected goals
        mu_mult    : float in [0.88, 1.12] — multiplier for away expected goals
        draw_boost : float in [0.0,  0.04] — addend to prob_draw (applied before normalisation)
        notes      : list[str] — human-readable descriptions of active conditions
    """
    if not weather:
        return _neutral()

    wind  = weather.get("wind_speed_kmh")
    rain  = weather.get("precipitation_mm")
    temp  = weather.get("temperature_c")
    outdoor = weather.get("is_outdoor", True)

    # Indoor stadiums: no weather impact
    if not outdoor:
        return _neutral()

    # Start from neutral
    goal_reduction = 0.0    # fraction to reduce both lam and mu
    draw_boost     = 0.0    # pp to add to draw probability
    notes          = []

    # ── Wind impact ──────────────────────────────────────────────────────
    # Wind degrades long passes, set pieces, and crossing precision.
    # Effect is symmetric: both teams affected equally.
    if wind is not None:
        if wind >= _WIND_VERY_STRONG:
            goal_reduction += 0.08
            draw_boost     += 0.025
            notes.append(f"Viento muy fuerte ({wind:.0f} km/h): -8% goles esperados")
        elif wind >= _WIND_STRONG:
            goal_reduction += 0.05
            draw_boost     += 0.015
            notes.append(f"Viento fuerte ({wind:.0f} km/h): -5% goles esperados")
        elif wind >= _WIND_MODERATE:
            goal_reduction += 0.02
            draw_boost     += 0.005
            notes.append(f"Viento moderado ({wind:.0f} km/h): -2% goles esperados")

    # ── Rain impact ───────────────────────────────────────────────────────
    # Heavy rain degrades pitch quality, slows play, reduces total goals.
    if rain is not None:
        if rain >= _RAIN_VERY_HEAVY:
            goal_reduction += 0.07
            draw_boost     += 0.020
            notes.append(f"Lluvia intensa ({rain:.1f} mm/h): -7% goles esperados")
        elif rain >= _RAIN_HEAVY:
            goal_reduction += 0.04
            draw_boost     += 0.012
            notes.append(f"Lluvia fuerte ({rain:.1f} mm/h): -4% goles esperados")
        elif rain >= _RAIN_LIGHT:
            goal_reduction += 0.01
            draw_boost     += 0.003
            notes.append(f"Lluvia ligera ({rain:.1f} mm/h): -1% goles esperados")

    # ── Temperature impact ────────────────────────────────────────────────
    # Very cold: muscle performance reduced, more cautious play.
    # Very hot: high-intensity pressing reduced after 60min (late goals fewer).
    if temp is not None:
        if temp <= _TEMP_VERY_COLD:
            goal_reduction += 0.03
            draw_boost     += 0.010
            notes.append(f"Temperatura bajo cero ({temp:.1f}°C): -3% goles esperados")
        elif temp <= _TEMP_COLD:
            goal_reduction += 0.01
            draw_boost     += 0.003
            notes.append(f"Temperatura fria ({temp:.1f}°C): -1% goles esperados")
        elif temp >= _TEMP_VERY_HOT:
            goal_reduction += 0.04
            draw_boost     += 0.015
            notes.append(f"Calor extremo ({temp:.1f}°C): -4% goles esperados")
        elif temp >= _TEMP_HOT:
            goal_reduction += 0.02
            draw_boost     += 0.007
            notes.append(f"Calor alto ({temp:.1f}°C): -2% goles esperados")

    # Cap total reduction at 12% (never exceed academic bounds)
    goal_reduction = min(goal_reduction, 0.12)
    draw_boost     = min(draw_boost,     0.04)

    # Only log if there's meaningful impact
    if goal_reduction < 0.01:
        notes = []
        draw_boost = 0.0

    mult = round(1.0 - goal_reduction, 4)
    return {
        "lam_mult":   mult,
        "mu_mult":    mult,      # symmetric — weather affects both teams equally
        "draw_boost": round(draw_boost, 4),
        "notes":      notes,
        "raw": {
            "wind_kmh":      wind,
            "rain_mm_h":     rain,
            "temperature_c": temp,
        },
    }


def _neutral() -> dict:
    return {
        "lam_mult":   1.0,
        "mu_mult":    1.0,
        "draw_boost": 0.0,
        "notes":      [],
        "raw":        {},
    }
