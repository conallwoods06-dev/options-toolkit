import numpy as np


def monte_carlo_price(S, K, T, r, sigma, n_sims=100_000, option_type="call", seed=None,
                      antithetic=False):
    """Price a European option by simulating many possible stock prices at expiry."""
    rng = np.random.default_rng(seed)                       # random number generator
    if antithetic:
        Z_half = rng.standard_normal(n_sims // 2)           # half the random numbers...
        Z = np.concatenate([Z_half, -Z_half])               # ...plus their mirror images
    else:
        Z = rng.standard_normal(n_sims)                     # n_sims random bell-curve numbers
    ST = S * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)   # stock prices at expiry
    if option_type == "call":
        payoffs = np.maximum(ST - K, 0)
    else:
        payoffs = np.maximum(K - ST, 0)
    discounted = np.exp(-r * T) * payoffs                   # each payoff in today's money
    if antithetic:
        half = len(Z_half)
        pairs = (discounted[:half] + discounted[half:]) / 2  # average each path with its mirror
        price = pairs.mean()
        std_error = pairs.std(ddof=1) / np.sqrt(half)       # pairs are the independent samples
    else:
        price = discounted.mean()                           # the average: our price estimate
        std_error = discounted.std(ddof=1) / np.sqrt(n_sims)  # how uncertain that estimate is
    return price, std_error