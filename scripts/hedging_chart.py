"""
Show where a delta-hedged option seller's profit comes from: implied vs realised volatility.
Run from the project folder with:
    python -m scripts.hedging_chart
"""

import matplotlib.pyplot as plt
import numpy as np

from optionskit.hedging import delta_hedge_pnl, simulate_paths

S, K, T, r = 100, 100, 1.0, 0.05
IMPLIED = 0.20          # the volatility the option is sold at
MU = 0.08               # the stock's real-world growth rate (doesn't matter: see the README)
N_PATHS = 20_000

fig, (ax_hist, ax_line, ax_freq) = plt.subplots(1, 3, figsize=(17, 5))

# 1. P&L distributions for three different realised volatilities
for realised, colour in [(0.15, "tab:green"), (0.20, "tab:grey"), (0.25, "tab:red")]:
    paths = simulate_paths(S, T, MU, realised, 252, N_PATHS, seed=1)
    pnl = delta_hedge_pnl(paths, K, T, r, IMPLIED)
    ax_hist.hist(pnl, bins=80, alpha=0.5, color=colour,
                 label=f"Realised {realised:.0%}: average €{round(pnl.mean(), 2) + 0:+.2f}")
ax_hist.axvline(0, color="black", linewidth=1)
ax_hist.set_title("P&L of a short call sold at 20% implied vol,\nhedged daily")
ax_hist.set_xlabel("Profit or loss per option (€)")
ax_hist.set_ylabel("Number of simulated paths")
ax_hist.legend()

# 2. Average P&L across a range of realised volatilities
realised_vols = np.linspace(0.10, 0.30, 11)
means = [delta_hedge_pnl(simulate_paths(S, T, MU, v, 252, N_PATHS, seed=1), K, T, r, IMPLIED).mean()
         for v in realised_vols]
ax_line.plot(realised_vols * 100, means, marker="o", color="tab:blue")
ax_line.axhline(0, color="black", linewidth=1)
ax_line.axvline(IMPLIED * 100, color="grey", linestyle="--", label="Implied vol (20%)")
ax_line.set_title("Average P&L vs realised volatility")
ax_line.set_xlabel("Realised volatility (%)")
ax_line.set_ylabel("Average profit per option (€)")
ax_line.grid(alpha=0.3)
ax_line.legend()

# 3. How hedging more often reduces the noise (realised = implied)
frequencies = [4, 12, 52, 252, 1000]
spreads = [delta_hedge_pnl(simulate_paths(S, T, MU, IMPLIED, n, N_PATHS, seed=1), K, T, r, IMPLIED).std()
           for n in frequencies]
ax_freq.plot(frequencies, spreads, marker="o", color="tab:purple")
ax_freq.set_xscale("log")
ax_freq.set_xticks(frequencies, ["Quarterly", "Monthly", "Weekly", "Daily", "4× daily"])
ax_freq.set_title("Hedging more often reduces P&L noise\n(realised vol = implied vol)")
ax_freq.set_xlabel("Hedging frequency")
ax_freq.set_ylabel("Standard deviation of P&L (€)")
ax_freq.grid(alpha=0.3)

fig.tight_layout()
fig.savefig("images/hedging.png", dpi=150)
print("Saved chart to images/hedging.png")