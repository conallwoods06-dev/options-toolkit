"""
Tests for the data cleaning and implied volatility steps. These use a made-up option chain,
so they run without an internet connection. Run from the project folder with:
    python -m pytest
"""

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from optionskit.black_scholes import call_price, put_price
from optionskit.market_data import add_implied_vols, clean_chain, time_to_expiry

S, T, r = 600, 0.1, 0.04


def true_vol(K):
    """A made-up skew: higher volatility for lower strikes, like the real market."""
    x = np.log(K / S)
    return 0.15 - 0.25 * x + 0.6 * x**2


def make_chain():
    """A fake option chain priced from the made-up skew, with a 2% bid-ask spread."""
    rows = []
    for K in range(450, 751, 10):
        for option_type, pricer in [("call", call_price), ("put", put_price)]:
            price = pricer(S, K, T, r, true_vol(K))
            rows.append({"strike": K, "option_type": option_type,
                         "bid": round(price * 0.99, 2), "ask": round(price * 1.01, 2) + 0.01,
                         "volume": 10, "openInterest": 100})
    return pd.DataFrame(rows)


def test_cleaning_keeps_only_out_of_the_money():
    df = clean_chain(make_chain(), S)
    assert (df[df["option_type"] == "put"]["strike"] < S).all()
    assert (df[df["option_type"] == "call"]["strike"] >= S).all()


def test_cleaning_removes_junk_quotes():
    junk = pd.DataFrame([
        {"strike": 650, "option_type": "call", "bid": 0, "ask": 1.0, "volume": 0, "openInterest": 0},     # no bid
        {"strike": 660, "option_type": "call", "bid": 1.0, "ask": 3.0, "volume": 5, "openInterest": 5},   # spread too wide
        {"strike": 700, "option_type": "call", "bid": 0.01, "ask": 0.02, "volume": 5, "openInterest": 5}, # too cheap
    ])
    assert len(clean_chain(junk, S)) == 0


def test_recovers_the_skew():
    """Implied vols from the fake prices should match the skew they were built from."""
    df = add_implied_vols(clean_chain(make_chain(), S), S, T, r)
    assert len(df) > 10
    assert np.allclose(df["iv"], true_vol(df["strike"]), atol=0.002)


def test_time_to_expiry():
    now = datetime(2026, 11, 16, 21, 0, tzinfo=timezone.utc)
    assert time_to_expiry("2026-12-16", now) == pytest.approx(30 / 365)



def test_atm_term_structure_reads_off_each_smile():
    """For each expiry, the at-the-money vol should be read off the smile where strike = spot."""
    from optionskit.market_data import atm_term_structure
    frames = []
    for days, atm in [(30, 0.14), (90, 0.16)]:
        frames.append(pd.DataFrame({"moneyness": [0.9, 1.0, 1.1], "iv": [atm + 0.05, atm, atm - 0.01],
                                    "expiry": f"E{days}", "T": days / 365}))
    terms = atm_term_structure(pd.concat(frames, ignore_index=True))
    assert list(terms["days"].round()) == [30, 90]
    assert np.allclose(terms["atm_iv"], [0.14, 0.16])