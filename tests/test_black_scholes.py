"""
Tests for the Black-Scholes pricer. Run from the project folder with:
    python -m pytest
"""

import numpy as np
import pytest

from optionskit.black_scholes import (
    call_price, put_price, delta, gamma, vega, theta, rho,
)

# A standard textbook case: at-the-money, 1 year, 5% rates, 20% vol
S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.2


def test_known_value():
    """Matches the widely published reference value for this textbook case."""
    assert call_price(S, K, T, r, sigma) == pytest.approx(10.4506, abs=1e-4)
    assert put_price(S, K, T, r, sigma) == pytest.approx(5.5735, abs=1e-4)


@pytest.mark.parametrize("strike", [80, 90, 100, 110, 120])
def test_put_call_parity(strike):
    """C - P = S - K*e^(-rT) must hold for any strike, or the prices allow free money."""
    lhs = call_price(S, strike, T, r, sigma) - put_price(S, strike, T, r, sigma)
    rhs = S - strike * np.exp(-r * T)
    assert lhs == pytest.approx(rhs, abs=1e-10)


def test_price_bounds():
    """A call is worth at least its discounted intrinsic value and at most the stock."""
    c = call_price(S, K, T, r, sigma)
    assert max(S - K * np.exp(-r * T), 0) <= c <= S


def test_higher_vol_means_higher_price():
    assert call_price(S, K, T, r, 0.3) > call_price(S, K, T, r, 0.2)


def _bump(f, arg_index, h=1e-4, **kwargs):
    """Numerical derivative of a pricing function by nudging one input up and down."""
    args = [S, K, T, r, sigma]
    up, down = list(args), list(args)
    up[arg_index] += h
    down[arg_index] -= h
    return (f(*up, **kwargs) - f(*down, **kwargs)) / (2 * h)


def test_greeks_match_finite_differences():
    """Each analytic Greek should match a brute-force numerical derivative."""
    assert delta(S, K, T, r, sigma) == pytest.approx(_bump(call_price, 0), rel=1e-5)
    assert vega(S, K, T, r, sigma) == pytest.approx(_bump(call_price, 4), rel=1e-5)
    assert rho(S, K, T, r, sigma) == pytest.approx(_bump(call_price, 3), rel=1e-5)
    # Theta is the change as time *passes*, i.e. as T decreases, hence the minus sign
    assert theta(S, K, T, r, sigma) == pytest.approx(-_bump(call_price, 2), rel=1e-5)
    # Gamma is the derivative of delta
    assert gamma(S, K, T, r, sigma) == pytest.approx(_bump(delta, 0), rel=1e-5)


def test_bad_inputs_raise():
    with pytest.raises(ValueError):
        call_price(S, K, 0, r, sigma)
    with pytest.raises(ValueError):
        call_price(S, K, T, r, -0.1)
