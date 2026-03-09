"""
Corners estimation (proxy model).

The free football-data.org tier doesn't provide corners data directly.
We estimate expected corners using total expected goals as a proxy:

  corners ≈ INTERCEPT + CORNERS_PER_GOAL_EQUIVALENT * (lambda + mu)

Where lambda and mu come from the Dixon-Coles model.
Coefficients are calibrated from published football analytics research.
"""

INTERCEPT = 2.5
CORNERS_PER_GOAL_EQUIVALENT = 2.8

# Threshold for Over X.5 lines
OVER_LINES = [8.5, 9.5, 10.5, 11.5]


def predict(lambda_: float, mu_: float) -> dict:
    """
    Estimate corner statistics for a match.

    Parameters
    ----------
    lambda_ : float
        Expected goals for home team (from Dixon-Coles).
    mu_ : float
        Expected goals for away team (from Dixon-Coles).

    Returns
    -------
    dict with:
        expected_corners     : float
        home_corners         : float (fraction of total)
        away_corners         : float
        over_lines           : dict {threshold: probability}
    """
    total_xg = lambda_ + mu_
    expected_corners = INTERCEPT + CORNERS_PER_GOAL_EQUIVALENT * total_xg

    # Home team generates more corners when stronger (proxy via xG share)
    h_share = lambda_ / total_xg if total_xg > 0 else 0.5
    home_corners = expected_corners * (0.45 + 0.20 * h_share)  # slight home bias
    away_corners = expected_corners - home_corners

    # Approximate over-line probabilities using a Poisson with expected total
    import math

    def poisson_cdf(k: float, lam: float) -> float:
        """P(X <= k) for Poisson(lam)."""
        cdf = 0.0
        for i in range(int(k) + 1):
            cdf += math.exp(-lam) * (lam ** i) / math.factorial(i)
        return cdf

    over_probs = {}
    for line in OVER_LINES:
        # P(corners > line) = 1 - P(corners <= floor(line))
        over_probs[line] = 1 - poisson_cdf(int(line), expected_corners)

    return {
        "expected_corners": expected_corners,
        "home_corners": home_corners,
        "away_corners": away_corners,
        "over_lines": over_probs,
    }
