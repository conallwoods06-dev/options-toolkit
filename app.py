"""
Interactive dashboard for the options toolkit. Run from the project folder with:
    streamlit run app.py
"""

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from optionskit.binomial import binomial_price
from optionskit.black_scholes import call_price, delta, gamma, put_price, rho, theta, vega
from optionskit.hedging import delta_hedge_pnl, simulate_paths
from optionskit.implied_vol import implied_volatility
from optionskit.market_data import atm_term_structure, get_expiries, load_smile, load_surface
from optionskit.monte_carlo import monte_carlo_price
from scripts.surface_chart import make_grid

st.set_page_config(page_title="Options Pricing & Volatility Toolkit", layout="wide")
st.title("Options Pricing & Volatility Surface Toolkit")
st.caption("Black-Scholes, binomial tree and Monte Carlo pricing, implied volatility from live "
           "market data, and delta-hedging simulation. Built in Python.")


# Cached data loaders: live downloads are slow, so results are kept for 15 minutes
@st.cache_data(ttl=900)
def cached_expiries(ticker):
    return get_expiries(ticker)


@st.cache_data(ttl=900)
def cached_smile(ticker, expiry):
    return load_smile(ticker, expiry)


@st.cache_data(ttl=900)
def cached_surface(ticker):
    return load_surface(ticker)


pricer_tab, iv_tab, smile_tab, surface_tab, hedge_tab = st.tabs(
    ["Option pricer", "Implied volatility", "Live smile", "Volatility surface", "Delta hedging"])


# ---------------------------------------------------------------------------
# Tab 1: price one option three ways
# ---------------------------------------------------------------------------
with pricer_tab:
    inputs, results = st.columns([1, 2])
    with inputs:
        option_type = st.selectbox("Option type", ["call", "put"])
        S = st.number_input("Stock price (S)", min_value=1.0, value=100.0, step=1.0)
        K = st.number_input("Strike (K)", min_value=1.0, value=100.0, step=1.0)
        days = st.number_input("Days to expiry", min_value=1, value=365, step=1)
        r = st.number_input("Interest rate (%)", value=5.0, step=0.25) / 100
        sigma = st.number_input("Volatility (%)", min_value=1.0, value=20.0, step=1.0) / 100
        T = days / 365

    with results:
        pricer = call_price if option_type == "call" else put_price
        bs = pricer(S, K, T, r, sigma)
        euro = binomial_price(S, K, T, r, sigma, option_type=option_type)
        amer = binomial_price(S, K, T, r, sigma, option_type=option_type, american=True)
        mc, se = monte_carlo_price(S, K, T, r, sigma, option_type=option_type, seed=42, antithetic=True)

        st.subheader("Price")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Black-Scholes", f"€{bs:.4f}")
        c2.metric("Binomial (European)", f"€{euro:.4f}")
        c3.metric("Binomial (American)", f"€{amer:.4f}", f"{amer - euro:+.4f} early exercise")
        c4.metric("Monte Carlo", f"€{mc:.4f}", f"± {1.96 * se:.4f} (95%)", delta_color="off")

        st.subheader("Greeks")
        g1, g2, g3, g4, g5 = st.columns(5)
        g1.metric("Delta", f"{delta(S, K, T, r, sigma, option_type):.4f}")
        g2.metric("Gamma", f"{gamma(S, K, T, r, sigma):.4f}")
        g3.metric("Vega (per 1% vol)", f"{vega(S, K, T, r, sigma) / 100:.4f}")
        g4.metric("Theta (per day)", f"{theta(S, K, T, r, sigma, option_type) / 365:.4f}")
        g5.metric("Rho (per 1% rate)", f"{rho(S, K, T, r, sigma, option_type) / 100:.4f}")

        stock_range = np.linspace(0.5 * S, 1.5 * S, 200)
        today_value = [pricer(s, K, T, r, sigma) for s in stock_range]
        payoff = np.maximum(stock_range - K, 0) if option_type == "call" else np.maximum(K - stock_range, 0)
        fig = go.Figure()
        fig.add_scatter(x=stock_range, y=today_value, name="Value today")
        fig.add_scatter(x=stock_range, y=payoff, name="Payoff at expiry", line={"dash": "dash"})
        fig.update_layout(title="Option value against stock price", xaxis_title="Stock price",
                          yaxis_title="Option value (€)", height=400)
        st.plotly_chart(fig)


# ---------------------------------------------------------------------------
# Tab 2: implied volatility from a price
# ---------------------------------------------------------------------------
with iv_tab:
    st.write("Enter a market price and find the volatility that makes Black-Scholes match it.")
    a, b = st.columns(2)
    with a:
        iv_type = st.selectbox("Option type ", ["call", "put"])
        iv_S = st.number_input("Stock price ", min_value=1.0, value=100.0)
        iv_K = st.number_input("Strike ", min_value=1.0, value=100.0)
    with b:
        iv_days = st.number_input("Days to expiry ", min_value=1, value=365)
        iv_r = st.number_input("Interest rate (%) ", value=5.0) / 100
        market_price = st.number_input("Market price of the option (€)", min_value=0.0, value=12.0)
    try:
        iv = implied_volatility(market_price, iv_S, iv_K, iv_days / 365, iv_r, iv_type)
        st.success(f"Implied volatility: **{iv:.2%}**")
    except ValueError as error:
        st.error(str(error))


# ---------------------------------------------------------------------------
# Tab 3: live volatility smile for one expiry
# ---------------------------------------------------------------------------
with smile_tab:
    ticker = st.text_input("Ticker", value="SPY").upper()
    try:
        expiries = cached_expiries(ticker)
    except Exception as error:
        expiries = []
        st.error(f"Couldn't load expiries for {ticker}: {error}")

    if expiries:
        expiry = st.selectbox("Expiry", expiries, index=min(4, len(expiries) - 1))
        if st.button("Load smile"):
            with st.spinner("Downloading option prices..."):
                try:
                    df, spot, rate, T_exp = cached_smile(ticker, expiry)
                except Exception as error:
                    st.error(f"Download failed: {error}")
                    df = None
            if df is not None and len(df) > 0:
                st.write(f"Spot **${spot:.2f}**, rate **{rate:.2%}**, **{T_exp * 365:.0f}** days to expiry, "
                         f"**{len(df)}** options after cleaning.")
                fig = go.Figure()
                for kind, colour in [("put", "crimson"), ("call", "royalblue")]:
                    part = df[df["option_type"] == kind]
                    fig.add_scatter(x=part["moneyness"] * 100, y=part["iv"] * 100, mode="markers",
                                    name=f"{kind}s", marker={"color": colour, "size": 6},
                                    text=part["strike"], hovertemplate="Strike %{text}<br>IV %{y:.2f}%")
                fig.add_vline(x=100, line_dash="dash", line_color="grey")
                fig.update_layout(title=f"{ticker} implied volatility smile, expiry {expiry}",
                                  xaxis_title="Strike as % of spot", yaxis_title="Implied volatility (%)",
                                  height=500)
                st.plotly_chart(fig)
                st.dataframe(df[["strike", "option_type", "bid", "ask", "mid", "iv"]], hide_index=True)


# ---------------------------------------------------------------------------
# Tab 4: full volatility surface
# ---------------------------------------------------------------------------
with surface_tab:
    surf_ticker = st.text_input("Ticker ", value="SPY").upper()
    st.caption("Downloads every expiry from 7 to 400 days out, so this can take up to a minute.")
    if st.button("Build surface"):
        with st.spinner("Downloading all expiries..."):
            try:
                surface, spot, rate = cached_surface(surf_ticker)
            except Exception as error:
                st.error(f"Download failed: {error}")
                surface = None
        if surface is not None and len(surface) > 0:
            m_grid, d_grid, iv_grid = make_grid(surface)
            fig = go.Figure(go.Surface(x=m_grid * 100, y=d_grid, z=iv_grid * 100,
                                       colorscale="Viridis", colorbar={"title": "IV (%)"}))
            fig.update_layout(title=f"{surf_ticker} implied volatility surface", height=650,
                              scene={"xaxis_title": "Strike as % of spot", "yaxis_title": "Days to expiry",
                                     "zaxis_title": "Implied volatility (%)"})
            st.plotly_chart(fig)

            terms = atm_term_structure(surface)
            fig = go.Figure(go.Scatter(x=terms["days"], y=terms["atm_iv"] * 100, mode="lines+markers"))
            fig.update_layout(title="At-the-money implied volatility by expiry", xaxis_title="Days to expiry",
                              yaxis_title="Implied volatility (%)", height=400)
            st.plotly_chart(fig)


# ---------------------------------------------------------------------------
# Tab 5: delta-hedging simulation
# ---------------------------------------------------------------------------
with hedge_tab:
    st.write("Sell an at-the-money call (S = K = 100, 1 year, 5% rates) at an implied volatility, "
             "delta-hedge it, and see how profit depends on the volatility that actually happens.")
    a, b = st.columns(2)
    with a:
        sold_at = st.slider("Implied volatility the option is sold at (%)", 5, 50, 20) / 100
        realised = st.slider("Realised volatility of the stock (%)", 5, 50, 20) / 100
    with b:
        frequency = st.selectbox("Hedging frequency", ["Daily", "Weekly", "Monthly"])
        n_paths = st.slider("Number of simulated paths", 1_000, 20_000, 5_000, step=1_000)
    n_steps = {"Daily": 252, "Weekly": 52, "Monthly": 12}[frequency]

    paths = simulate_paths(100, 1.0, 0.08, realised, n_steps, n_paths, seed=1)
    pnl = delta_hedge_pnl(paths, 100, 1.0, 0.05, sold_at)
    rule_of_thumb = vega(100, 100, 1.0, 0.05, sold_at) * (sold_at - realised)

    m1, m2, m3 = st.columns(3)
    m1.metric("Average P&L", f"€{pnl.mean():+.2f}")
    m2.metric("Rule of thumb: vega × (implied − realised)", f"€{rule_of_thumb:+.2f}")
    m3.metric("Spread of outcomes (std dev)", f"€{pnl.std():.2f}")

    fig = go.Figure(go.Histogram(x=pnl, nbinsx=80))
    fig.add_vline(x=0, line_color="black")
    fig.update_layout(title="Distribution of profit and loss across simulated paths",
                      xaxis_title="Profit or loss per option (€)", yaxis_title="Number of paths", height=450)
    st.plotly_chart(fig)
