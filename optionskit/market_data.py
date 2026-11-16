"""
Fetch real option prices with yfinance, clean them, and calculate implied volatilities.
"""

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from optionskit.implied_vol import implied_volatility


def get_spot_price(ticker):
    """Latest closing price of the underlying stock or ETF."""
    import yfinance as yf
    return float(yf.Ticker(ticker).history(period="5d")["Close"].iloc[-1])


def get_risk_free_rate():
    """US 13-week Treasury bill yield (^IRX), as a decimal. Falls back to 4% if unavailable."""
    import yfinance as yf
    try:
        return float(yf.Ticker("^IRX").history(period="5d")["Close"].iloc[-1]) / 100
    except Exception:
        return 0.04


def get_expiries(ticker):
    """All available expiry dates, as strings like '2026-12-18'."""
    import yfinance as yf
    return list(yf.Ticker(ticker).options)


def time_to_expiry(expiry, now=None):
    """Years until an expiry date, assuming options expire at 4pm New York time (21:00 UTC)."""
    now = now or datetime.now(timezone.utc)
    expiry_time = datetime.strptime(expiry, "%Y-%m-%d").replace(hour=21, tzinfo=timezone.utc)
    return (expiry_time - now).total_seconds() / (365 * 24 * 3600)


def get_raw_chain(ticker, expiry):
    """Download calls and puts for one expiry and stack them into one table."""
    import yfinance as yf
    chain = yf.Ticker(ticker).option_chain(expiry)
    calls = chain.calls.assign(option_type="call")
    puts = chain.puts.assign(option_type="put")
    return pd.concat([calls, puts], ignore_index=True)


def clean_chain(raw, S, max_spread=0.25, min_price=0.05):
    """Keep only options whose prices are reliable enough to calculate implied volatility from.

    - both a bid and an ask (someone is actually willing to trade)
    - a bid-ask spread no wider than max_spread (25%) of the mid-price
    - a mid-price of at least min_price (5 cents)
    - out of the money only: puts below the spot price, calls above it
    """
    df = raw[["strike", "option_type", "bid", "ask", "volume", "openInterest"]].copy()
    df["mid"] = (df["bid"] + df["ask"]) / 2
    df["spread"] = (df["ask"] - df["bid"]) / df["mid"]

    has_quotes = (df["bid"] > 0) & (df["ask"] > df["bid"])
    tight = df["spread"] <= max_spread
    not_tiny = df["mid"] >= min_price
    otm = ((df["option_type"] == "call") & (df["strike"] >= S)) | \
          ((df["option_type"] == "put") & (df["strike"] < S))

    return df[has_quotes & tight & not_tiny & otm].sort_values("strike").reset_index(drop=True)


def add_implied_vols(df, S, T, r):
    """Calculate implied volatility for every row. Rows the solver rejects get NaN and are dropped."""
    def solve(row):
        try:
            return implied_volatility(row["mid"], S, row["strike"], T, r, row["option_type"])
        except (ValueError, RuntimeError):
            return np.nan

    df = df.copy()
    df["iv"] = df.apply(solve, axis=1)
    return df.dropna(subset=["iv"]).reset_index(drop=True)


def load_smile(ticker, expiry, S=None, r=None):
    """Everything in one go: download, clean and calculate implied volatilities for one expiry."""
    S = S if S is not None else get_spot_price(ticker)
    r = r if r is not None else get_risk_free_rate()
    T = time_to_expiry(expiry)
    df = add_implied_vols(clean_chain(get_raw_chain(ticker, expiry), S), S, T, r)
    df["moneyness"] = df["strike"] / S
    df["expiry"] = expiry
    df["T"] = T
    return df, S, r, T