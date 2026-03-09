"""
Cards estimation (proxy model).

The free football-data.org tier doesn't provide card data in match results,
so this is a display-only model — it is NOT tracked for accuracy.

Expected yellow cards are estimated from Dixon-Coles lambda/mu using a
linear proxy calibrated on top-5 league averages (~3.8 yellows/game):

  expected_cards ≈ INTERCEPT + CARDS_PER_XG * (lambda + mu)

Research shows a slight positive correlation between match openness (xG)
and card volume (intense, close games produce more cards). The home team
typically receives slightly fewer cards due to referee home bias.

Over lines commonly offered by bookmakers: 3.5 and 4.5.
"""

INTERCEPT = 2.9
CARDS_PER_XG = 0.38

# Threshold for Over X.5 lines
OVER_LINES = [3.5, 4.5]

# Home teams receive ~10% fewer cards on average (referee home bias)
HOME_BIAS = 0.46  # fraction of total cards going to home team


def predict(lambda_: float, mu_: float) -> dict:
    """
    Estimate yellow card statistics for a match.

    Parameters
    ----------
    lambda_ : float
        Expected goals for home team (from Dixon-Coles).
    mu_ : float
        Expected goals for away team (from Dixon-Coles).

    Returns
    -------
    dict with:
        expected_cards : float  — total expected yellow cards
        home_cards     : float  — expected home yellow cards
        away_cards     : float  — expected away yellow cards
        over_lines     : dict {threshold: probability}
    """
    import math

    total_xg = lambda_ + mu_
    expected_cards = INTERCEPT + CARDS_PER_XG * total_xg

    # Home team typically receives fewer cards (referee home bias)
    # Adjust slightly based on attacking strength (weaker side gets pressed more)
    h_share = lambda_ / total_xg if total_xg > 0 else 0.5
    # Weaker attacking team (lower xG share) gets more cards defending
    home_cards = expected_cards * (HOME_BIAS - 0.06 * (h_share - 0.5))
    away_cards = expected_cards - home_cards

    def poisson_cdf(k: float, lam: float) -> float:
        """P(X <= k) for Poisson(lam)."""
        cdf = 0.0
        for i in range(int(k) + 1):
            cdf += math.exp(-lam) * (lam ** i) / math.factorial(i)
        return cdf

    over_probs = {}
    for line in OVER_LINES:
        over_probs[line] = 1 - poisson_cdf(int(line), expected_cards)

    return {
        "expected_cards": expected_cards,
        "home_cards": home_cards,
        "away_cards": away_cards,
        "over_lines": over_probs,
    }
