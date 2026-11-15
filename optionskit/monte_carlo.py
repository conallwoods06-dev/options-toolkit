import numpy as np


def monte_carlo_price(S, K, T, r, sigma, n_sims=100_000, option_type="call", seed=None):
    """Price a European option by simulating many possible stock prices at expiry."""
    rng = np.random.default_rng(seed)                       # random number generator
    Z = rng.standard_normal(n_sims)                         # n_sims random bell-curve numbers
    ST = S * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)   # stock prices at expiry
    if option_type == "call":
        payoffs = np.maximum(ST - K, 0)
    else:
        payoffs = np.maximum(K - ST, 0)
    discounted = np.exp(-r * T) * payoffs                   # each payoff in today's money
    price = discounted.mean()                               # the average: our price estimate
    std_error = discounted.std(ddof=1) / np.sqrt(n_sims)    # how uncertain that estimate is
    return price, std_error                 # average payoff, discounted to today