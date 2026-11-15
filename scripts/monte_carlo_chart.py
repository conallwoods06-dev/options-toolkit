"""
Plot how the Monte Carlo estimate and its 95% confidence band narrow towards Black-Scholes
as the number of simulations grows. Run from the project folder with:
    python -m scripts.monte_carlo_chart
"""

import matplotlib.pyplot as plt
import numpy as np

from optionskit.black_scholes import call_price
from optionskit.monte_carlo import monte_carlo_price

S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.2

sim_counts = np.logspace(2, 6, 30).astype(int)      # 30 values from 100 to 1,000,000
prices, errors = [], []
for n in sim_counts:
    price, se = monte_carlo_price(S, K, T, r, sigma, n_sims=n, seed=42)
    prices.append(price)
    errors.append(se)

prices = np.array(prices)
errors = np.array(errors)
bs_price = call_price(S, K, T, r, sigma)

fig, ax = plt.subplots(figsize=(9, 5))
ax.fill_between(sim_counts, prices - 1.96 * errors, prices + 1.96 * errors,
                color="tab:blue", alpha=0.2, label="95% confidence interval")
ax.plot(sim_counts, prices, color="tab:blue", marker="o", markersize=3,
        linewidth=1.2, label="Monte Carlo estimate")
ax.axhline(bs_price, color="tab:red", linestyle="--", linewidth=1.5,
           label=f"Black-Scholes (€{bs_price:.2f})")

ax.set_xscale("log")
ax.set_title("Monte Carlo estimate converges to Black-Scholes\n"
             "European call: S=100, K=100, T=1 year, r=5%, σ=20%")
ax.set_xlabel("Number of simulations (log scale)")
ax.set_ylabel("Option price (€)")
ax.grid(alpha=0.3, which="both")
ax.legend()

fig.tight_layout()
fig.savefig("images/monte_carlo.png", dpi=150)
print("Saved chart to images/monte_carlo.png")