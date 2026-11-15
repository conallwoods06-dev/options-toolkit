import numpy as np


def binomial_price(S, K, T, r, sigma, N=500, option_type="call", american=False):
    """Price an option with a Cox-Ross-Rubinstein binomial tree."""
    dt = T / N                                  # length of one time step
    u = np.exp(sigma * np.sqrt(dt))             # up factor
    d = 1 / u                                   # down factor
    p = (np.exp(r * dt) - d) / (u - d)          # risk-neutral probability of an up move
    j = np.arange(N + 1)                        # number of up moves: 0, 1, ..., N
    prices = S * u**j * d**(N - j)              # every possible stock price at expiry
    if option_type == "call":
        payoffs = np.maximum(prices - K, 0)      # option value at expiry (changes every negative to 0 as if negative you do not exercise option to buy)  
    else:
        payoffs = np.maximum(K - prices, 0)           # option value at expiry
    disc = np.exp(-r * dt)                      # discount factor for one step
    for _ in range(N):                          # step back N times
        payoffs = disc * (p * payoffs[1:] + (1 - p) * payoffs[:-1])   # value of holding on
        if american:
            prices = prices[:-1] * u                                  # stock prices one step earlier
            if option_type == "call":
                exercise = np.maximum(prices - K, 0)                  # value of exercising now
            else:
                exercise = np.maximum(K - prices, 0)
            payoffs = np.maximum(payoffs, exercise)                   # holder picks the better one
    return payoffs[0]