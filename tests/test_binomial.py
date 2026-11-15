"""
Tests for the binomial tree pricer. Run from the project folder with:
    python -m pytest
"""

import numpy as np
import pytest

from optionskit.binomial import binomial_price
from optionskit.black_scholes import call_price, put_price

# The same textbook case used in the Black-Scholes tests
S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.2


def test_matches_hand_worked_example():
    """With 3 steps, the tree gives the values worked out by hand."""
    assert binomial_price(S, K, T, r, sigma, N=3) == pytest.approx(11.0439, abs=1e-4)
    assert binomial_price(S, K, T, r, sigma, N=3, option_type="put") == pytest.approx(6.1668, abs=1e-4)
    assert binomial_price(S, K, T, r, sigma, N=3, option_type="put", american=True) == pytest.approx(6.4996, abs=1e-4)


@pytest.mark.parametrize("strike", [80, 100, 120])
def test_european_converges_to_black_scholes(strike):
    """With many steps, the tree should agree with Black-Scholes to within a cent."""
    assert binomial_price(S, strike, T, r, sigma) == pytest.approx(call_price(S, strike, T, r, sigma), abs=0.01)
    assert binomial_price(S, strike, T, r, sigma, option_type="put") == pytest.approx(put_price(S, strike, T, r, sigma), abs=0.01)


def test_put_call_parity():
    """C - P = S - K*e^(-rT) holds exactly for European options in the tree too."""
    c = binomial_price(S, K, T, r, sigma, N=200)
    p = binomial_price(S, K, T, r, sigma, N=200, option_type="put")
    assert c - p == pytest.approx(S - K * np.exp(-r * T), abs=1e-8)


def test_american_call_equals_european_call():
    """With no dividends, it is never worth exercising a call early."""
    european = binomial_price(S, K, T, r, sigma)
    american = binomial_price(S, K, T, r, sigma, american=True)
    assert american == pytest.approx(european, abs=1e-10)


def test_american_put_worth_more_than_european():
    """The right to exercise early is worth something for a put."""
    european = binomial_price(S, K, T, r, sigma, option_type="put")
    american = binomial_price(S, K, T, r, sigma, option_type="put", american=True)
    assert american > european


def test_american_put_at_least_intrinsic_value():
    """An American option can be exercised now, so it is worth at least what that pays."""
    deep_in_the_money = binomial_price(60, K, T, r, sigma, option_type="put", american=True)
    assert deep_in_the_money >= K - 60 - 1e-9   # tiny allowance for computer rounding