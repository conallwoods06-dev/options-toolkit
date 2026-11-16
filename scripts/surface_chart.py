"""
Build the implied volatility surface from live option prices, across all strikes and expiries.
Run from the project folder with, for example:
    python -m scripts.surface_chart SPY

Saves:
    data/surface_SPY_<date>.csv          the cleaned data, so results can be reproduced later
    images/term_structure_SPY.png        at-the-money implied volatility against expiry
    images/surface_SPY.png               a 3D picture of the surface (for the README)
    images/surface_SPY.html              an interactive 3D surface you can rotate in a browser
"""

import sys
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata

from optionskit.market_data import atm_term_structure, load_surface


def plot_term_structure(terms, ticker, filename):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(terms["days"], terms["atm_iv"] * 100, marker="o", color="tab:blue")
    ax.set_title(f"{ticker} at-the-money implied volatility by expiry — data from {date.today()}")
    ax.set_xlabel("Days to expiry")
    ax.set_ylabel("At-the-money implied volatility (%)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    print(f"Saved chart to {filename}")


def make_grid(surface, m_low=0.80, m_high=1.15):
    """Spread the scattered (moneyness, days, iv) points onto a regular grid so they can be drawn as a surface."""
    pts = surface[(surface["moneyness"] >= m_low) & (surface["moneyness"] <= m_high)]
    days = pts["T"] * 365
    m_grid, d_grid = np.meshgrid(np.linspace(m_low, m_high, 60),
                                 np.linspace(days.min(), days.max(), 60))
    iv_grid = griddata((pts["moneyness"], days), pts["iv"], (m_grid, d_grid), method="linear")
    return m_grid, d_grid, iv_grid


def plot_surface_png(m_grid, d_grid, iv_grid, ticker, filename):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(projection="3d")
    ax.plot_surface(m_grid * 100, d_grid, np.ma.masked_invalid(iv_grid * 100),
                    cmap="viridis", edgecolor="none")
    ax.set_title(f"{ticker} implied volatility surface — data from {date.today()}")
    ax.set_xlabel("Strike as % of spot")
    ax.set_ylabel("Days to expiry")
    ax.set_zlabel("Implied volatility (%)")
    ax.view_init(elev=25, azim=-50)
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    print(f"Saved chart to {filename}")


def plot_surface_html(m_grid, d_grid, iv_grid, ticker, filename):
    import plotly.graph_objects as go
    fig = go.Figure(go.Surface(x=m_grid * 100, y=d_grid, z=iv_grid * 100, colorscale="Viridis",
                               colorbar={"title": "IV (%)"}))
    fig.update_layout(title=f"{ticker} implied volatility surface — data from {date.today()}",
                      scene={"xaxis_title": "Strike as % of spot",
                             "yaxis_title": "Days to expiry",
                             "zaxis_title": "Implied volatility (%)"})
    fig.write_html(filename)
    print(f"Saved interactive chart to {filename}")


if __name__ == "__main__":
    ticker = sys.argv[1]
    surface, S, r = load_surface(ticker)
    print(f"{ticker}: spot ${S:.2f}, rate {r:.2%}, {len(surface)} options across "
          f"{surface['expiry'].nunique()} expiries")

    Path("data").mkdir(exist_ok=True)
    surface.to_csv(f"data/surface_{ticker}_{date.today()}.csv", index=False)

    plot_term_structure(atm_term_structure(surface), ticker, f"images/term_structure_{ticker}.png")
    grids = make_grid(surface)
    plot_surface_png(*grids, ticker, f"images/surface_{ticker}.png")
    plot_surface_html(*grids, ticker, f"images/surface_{ticker}.html")