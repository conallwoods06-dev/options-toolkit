import numpy as np
from scipy.optimize import brentq

from optionskit.black_scholes import call_price, put_price, vega

SIGMA_LOW, SIGMA_HIGH = 1e-6, 5.0                           # search between ~0% and 500% volatility


def implied_volatility(price, S, K, T, r, option_type="call", tol=1e-8, max_iter=100):
    """Find the volatility that makes Black-Scholes match a market price.

    Tries Newton-Raphson first (fast), and falls back to Brent's method (slower,
    but guaranteed to work) if Newton-Raphson goes off track.
    """
    pricer = call_price if option_type == "call" else put_price

    # 1. Check the price is possible at all
    if option_type == "call":
        lower, upper = max(S - K * np.exp(-r * T), 0), S
    else:
        lower, upper = max(K * np.exp(-r * T) - S, 0), K * np.exp(-r * T)
    if not lower < price < upper:
        raise ValueError(f"Price {price} is outside the possible range ({lower:.4f}, {upper:.4f}).")

    # 2. Try Newton-Raphson
    sigma = 0.2                                             # starting guess: 20%
    for _ in range(max_iter):
        diff = pricer(S, K, T, r, sigma) - price            # how far off our guess is
        if abs(diff) < tol:                                 # close enough: done
            return sigma
        v = vega(S, K, T, r, sigma)
        if v < 1e-8:                                        # vega too small: step would be huge
            break
        sigma = sigma - diff / v                            # Newton-Raphson step
        if not SIGMA_LOW < sigma < SIGMA_HIGH:              # jumped somewhere silly
            break

    # 3. Fall back to Brent's method
    return brentq(lambda s: pricer(S, K, T, r, s) - price, SIGMA_LOW, SIGMA_HIGH, xtol=tol)