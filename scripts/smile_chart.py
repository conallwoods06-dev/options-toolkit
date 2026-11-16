"""
Plot the implied volatility smile for one expiry, using live option prices.
Run from the project folder with, for example:
    python -m scripts.smile_chart SPY 2026-12-18
"""

import sys
from datetime import date

import matplotlib.pyplot as plt

from optionskit.market_data import load_smile


def plot_smile(df, ticker, expiry, S, T, filename):
    near_the_money = df[(df["moneyness"] > 0.85) & (df["moneyness"] < 1.10)]

    fig, (ax_full, ax_zoom) = plt.subplots(1, 2, figsize=(13, 5))
    for ax, title, data in [(ax_full, "All cleaned strikes", df),
                            (ax_zoom, "Near the money (85%–110% of spot)", near_the_money)]:
        puts = data[data["option_type"] == "put"]
        calls = data[data["option_type"] == "call"]
        ax.scatter(puts["moneyness"] * 100, puts["iv"] * 100, s=12, color="tab:red", label="Puts (below spot)")
        ax.scatter(calls["moneyness"] * 100, calls["iv"] * 100, s=12, color="tab:blue", label="Calls (above spot)")
        ax.axvline(100, color="grey", linestyle="--", linewidth=1, label=f"Spot (${S:.2f})")
        ax.set_title(title)
        ax.set_xlabel("Strike as % of spot price")
        ax.set_ylabel("Implied volatility (%)")
        ax.grid(alpha=0.3)
        ax.legend()

    fig.suptitle(f"{ticker} implied volatility smile — expiry {expiry} "
                 f"({T * 365:.0f} days), data from {date.today()}")
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    print(f"Saved chart to {filename}")


if __name__ == "__main__":
    ticker, expiry = sys.argv[1], sys.argv[2]
    df, S, r, T = load_smile(ticker, expiry)
    print(f"{ticker}: spot ${S:.2f}, rate {r:.2%}, {len(df)} options after cleaning")
    plot_smile(df, ticker, expiry, S, T, f"images/smile_{ticker}_{expiry}.png")