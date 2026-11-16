"""
Tests for the delta-hedging simulation. Run from the project folder with:
    python -m pytest
"""

import numpy as np
import pytest

from optionskit.black_scholes import vega
from optionskit.hedging import delta_hedge_pnl, simulate_paths

S, K, T, r = 100, 100, 1.0, 0.05
IMPLIED = 0.20


def average_pnl(realised, mu=0.08, n_steps=252, option_type="call"):
    paths = simulate_paths(S, T, mu, realised, n_steps, 20_000, seed=1)
    return delta_hedge_pnl(paths, K, T, r, IMPLIED, option_type)


def test_paths_start_at_spot_and_have_right_shape():
    paths = simulate_paths(S, T, 0.05, 0.2, n_steps=10, n_paths=5, seed=1)
    assert paths.shape == (5, 11)
    assert np.all(paths[:, 0] == S)


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_breaks_even_when_realised_equals_implied(option_type):
    """If the stock moves exactly as much as the option was priced for, the hedge breaks even on average."""
    assert average_pnl(IMPLIED, option_type=option_type).mean() == pytest.approx(0, abs=0.05)


def test_seller_profits_when_realised_below_implied():
    assert average_pnl(0.15).mean() > 1


def test_seller_loses_when_realised_above_implied():
    assert average_pnl(0.25).mean() < -1


def test_profit_roughly_equals_vega_times_vol_difference():
    """Rule of thumb: P&L ≈ vega × (implied vol − realised vol)."""
    expected = vega(S, K, T, r, IMPLIED) * (IMPLIED - 0.15)
    assert average_pnl(0.15).mean() == pytest.approx(expected, rel=0.1)


def test_stock_growth_rate_does_not_matter():
    """The hedge removes exposure to the stock's direction, so its expected growth rate is irrelevant."""
    assert average_pnl(IMPLIED, mu=-0.10).mean() == pytest.approx(average_pnl(IMPLIED, mu=0.20).mean(), abs=0.05)


def test_hedging_more_often_reduces_noise():
    """About 5x more rebalancing should cut the spread of outcomes by about √5 ≈ 2.2x."""
    weekly = average_pnl(IMPLIED, n_steps=52).std()
    daily = average_pnl(IMPLIED, n_steps=252).std()
    assert weekly / daily == pytest.approx(np.sqrt(252 / 52), rel=0.15)