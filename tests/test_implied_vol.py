"""
Tests for the implied volatility solver. Run from the project folder with:
    python -m pytest
"""

import numpy as np
import pytest

from optionskit.black_scholes import call_price, put_price
from optionskit.implied_vol import implied_volatility

S, r = 100, 0.05


def test_worked_example():
    """A €12 at-the-money call implies about 24.11% volatility."""
    assert implied_volatility(12, S, 100, 1, r) == pytest.approx(0.241117, abs=1e-6)


@pytest.mark.parametrize("option_type", ["call", "put"])
@pytest.mark.parametrize("K", [70, 90, 100, 110, 130])
@pytest.mark.parametrize("T", [0.1, 0.5, 2.0])
@pytest.mark.parametrize("sigma", [0.1, 0.3, 0.8])
def test_round_trip(option_type, K, T, sigma):
    """Price an option at a known volatility, then check the solver recovers that volatility."""
    pricer = call_price if option_type == "call" else put_price
    price = pricer(S, K, T, r, sigma)
    if option_type == "call":
        intrinsic = max(S - K * np.exp(-r * T), 0)
    else:
        intrinsic = max(K * np.exp(-r * T) - S, 0)
    if price - intrinsic < 0.01:
        pytest.skip("Less than 1 cent of time value: volatility can't be recovered reliably.")
    assert implied_volatility(price, S, K, T, r, option_type) == pytest.approx(sigma, abs=1e-5)


def test_hard_case_needs_fallback():
    """Deep out-of-the-money and short-dated: plain Newton-Raphson fails here."""
    price = call_price(S, 150, 0.1, r, 0.6)
    assert implied_volatility(price, S, 150, 0.1, r) == pytest.approx(0.6, abs=1e-6)


@pytest.mark.parametrize("bad_price", [-1, 0, 100, 120])
def test_impossible_call_price_raises(bad_price):
    """No volatility can produce these prices, so the solver should refuse clearly."""
    with pytest.raises(ValueError):
        implied_volatility(bad_price, S, 100, 1, r)


def test_price_below_intrinsic_value_raises():
    """A deep in-the-money call can't be worth less than what exercising it would pay."""
    with pytest.raises(ValueError):
        implied_volatility(30, S, 60, 1, r)