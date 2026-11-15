"""
Black-Scholes pricing for European options.

Inputs used throughout:
    S     : current price of the underlying stock
    K     : strike price
    T     : time to expiry, in years (e.g. 0.5 = six months)
    r     : risk-free interest rate, continuously compounded (e.g. 0.04 = 4%)
    sigma : volatility, annualised (e.g. 0.2 = 20%)
"""

import numpy as np
from scipy.stats import norm


def _d1_d2(S, K, T, r, sigma):
    """The two intermediate quantities that appear in every Black-Scholes formula."""
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2


def _check_inputs(S, K, T, sigma):
    if S <= 0 or K <= 0:
        raise ValueError("Stock price and strike must be positive.")
    if T <= 0:
        raise ValueError("Time to expiry must be positive.")
    if sigma <= 0:
        raise ValueError("Volatility must be positive.")


def call_price(S, K, T, r, sigma):
    """Price of a European call: C = S*N(d1) - K*e^(-rT)*N(d2)."""
    _check_inputs(S, K, T, sigma)
    d1, d2 = _d1_d2(S, K, T, r, sigma)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def put_price(S, K, T, r, sigma):
    """Price of a European put: P = K*e^(-rT)*N(-d2) - S*N(-d1)."""
    _check_inputs(S, K, T, sigma)
    d1, d2 = _d1_d2(S, K, T, r, sigma)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def prob_in_the_money(S, K, T, r, sigma, option_type="call"):
    """Risk-neutral probability the option finishes in the money."""
    _check_inputs(S, K, T, sigma)
    _, d2 = _d1_d2(S, K, T, r, sigma)
    if option_type == "call":
        return norm.cdf(d2)
    return norm.cdf(-d2)


# ---------------------------------------------------------------------------
# The Greeks: how the option price responds to changes in each input
# ---------------------------------------------------------------------------

def delta(S, K, T, r, sigma, option_type="call"):
    """Change in option price per 1-unit move in the stock price."""
    d1, _ = _d1_d2(S, K, T, r, sigma)
    if option_type == "call":
        return norm.cdf(d1)
    return norm.cdf(d1) - 1


def gamma(S, K, T, r, sigma):
    """Rate of change of delta. Same for calls and puts."""
    d1, _ = _d1_d2(S, K, T, r, sigma)
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))


def vega(S, K, T, r, sigma):
    """Change in price per 1.00 (i.e. 100 percentage-point) change in volatility.
    Divide by 100 for 'per 1% vol'. Same for calls and puts."""
    d1, _ = _d1_d2(S, K, T, r, sigma)
    return S * norm.pdf(d1) * np.sqrt(T)


def theta(S, K, T, r, sigma, option_type="call"):
    """Change in price per year of passing time (usually negative).
    Divide by 365 for 'per day'."""
    d1, d2 = _d1_d2(S, K, T, r, sigma)
    time_decay = -S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
    if option_type == "call":
        return time_decay - r * K * np.exp(-r * T) * norm.cdf(d2)
    return time_decay + r * K * np.exp(-r * T) * norm.cdf(-d2)


def rho(S, K, T, r, sigma, option_type="call"):
    """Change in price per 1.00 (100 percentage-point) change in interest rates."""
    _, d2 = _d1_d2(S, K, T, r, sigma)
    if option_type == "call":
        return K * T * np.exp(-r * T) * norm.cdf(d2)
    return -K * T * np.exp(-r * T) * norm.cdf(-d2)


if __name__ == "__main__":
    # Quick demo: run `python -m optionskit.black_scholes` from the project folder.
    S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.2
    print(f"Call price: {call_price(S, K, T, r, sigma):.4f}")
    print(f"Put price:  {put_price(S, K, T, r, sigma):.4f}")
    print(f"Call delta: {delta(S, K, T, r, sigma, 'call'):.4f}")
    print(f"Gamma:      {gamma(S, K, T, r, sigma):.4f}")
    print(f"Vega:       {vega(S, K, T, r, sigma):.4f}")
