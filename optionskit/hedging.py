"""
Simulate selling an option and delta-hedging it, to see where an options trader's profit comes from.
"""

import numpy as np

from optionskit.black_scholes import call_price, put_price, delta


def simulate_paths(S0, T, mu, sigma, n_steps, n_paths, seed=None):
    """Simulate stock price paths. Returns a table: one row per path, one column per time step.

    mu is the stock's real-world average growth rate; sigma is its actual (realised) volatility.
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    Z = rng.standard_normal((n_paths, n_steps))
    log_returns = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
    log_paths = np.cumsum(log_returns, axis=1)                       # add up the moves along each path
    return S0 * np.exp(np.hstack([np.zeros((n_paths, 1)), log_paths]))  # start every path at S0


def delta_hedge_pnl(paths, K, T, r, sigma_implied, option_type="call"):
    """Profit or loss from selling one option at implied volatility and delta-hedging it until expiry.

    At each step the trader holds exactly 'delta' shares, so small stock moves cancel out.
    Cash earns (or pays) interest at rate r. Returns one P&L number per path.
    """
    n_steps = paths.shape[1] - 1
    dt = T / n_steps
    pricer = call_price if option_type == "call" else put_price
    S0 = paths[0, 0]

    # Day 0: sell the option, buy delta shares, keep the rest as cash
    premium = pricer(S0, K, T, r, sigma_implied)
    shares = np.full(paths.shape[0], delta(S0, K, T, r, sigma_implied, option_type))
    cash = premium - shares * S0

    # Each step: earn interest, then rebalance to the new delta
    for i in range(1, n_steps):
        cash = cash * np.exp(r * dt)
        S = paths[:, i]
        new_shares = delta(S, K, T - i * dt, r, sigma_implied, option_type)
        cash = cash - (new_shares - shares) * S                      # buy (or sell) the difference
        shares = new_shares

    # Expiry: last interest payment, then pay out on the option and value the shares held
    cash = cash * np.exp(r * dt)
    S_T = paths[:, -1]
    payoff = np.maximum(S_T - K, 0) if option_type == "call" else np.maximum(K - S_T, 0)
    return cash + shares * S_T - payoff