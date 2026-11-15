"""
Plot how the binomial tree price converges to Black-Scholes as the number of steps grows.
Run from the project folder with:
    python -m scripts.convergence_chart
"""

import matplotlib.pyplot as plt

from optionskit.binomial import binomial_price
from optionskit.black_scholes import call_price

S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.2

steps = list(range(1, 201))
tree_prices = [binomial_price(S, K, T, r, sigma, N=n) for n in steps]
bs_price = call_price(S, K, T, r, sigma)

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(steps, tree_prices, color="tab:blue", linewidth=1.2, label="Binomial tree")
ax.axhline(bs_price, color="tab:red", linestyle="--", linewidth=1.5,
           label=f"Black-Scholes (€{bs_price:.2f})")

ax.set_title("Binomial tree price converges to Black-Scholes\n"
             "European call: S=100, K=100, T=1 year, r=5%, σ=20%")
ax.set_xlabel("Number of steps in the tree (N)")
ax.set_ylabel("Option price (€)")
ax.set_ylim(bs_price - 1, bs_price + 1)
ax.grid(alpha=0.3)
ax.legend()

fig.tight_layout()
fig.savefig("images/convergence.png", dpi=150)
print("Saved chart to images/convergence.png")