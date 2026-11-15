"""
Tests for the Monte Carlo pricer. Run from the project folder with:
    python -m pytest
"""

import numpy as np
import pytest

from optionskit.monte_carlo import monte_carlo_price
from optionskit.black_scholes import call_price, put_price

S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.2


@pytest.mark.parametrize("strike", [80, 100, 120])
def test_agrees_with_black_scholes(strike):
    """The estimate should land within 3 standard errors of the exact price."""
    price, se = monte_carlo_price(S, strike, T, r, sigma, seed=42)
    assert abs(price - call_price(S, strike, T, r, sigma)) < 3 * se

    price, se = monte_carlo_price(S, strike, T, r, sigma, option_type="put", seed=42)
    assert abs(price - put_price(S, strike, T, r, sigma)) < 3 * se


def test_same_seed_gives_same_answer():
    """A fixed seed makes the 'random' numbers repeatable."""
    first = monte_carlo_price(S, K, T, r, sigma, seed=7)
    second = monte_carlo_price(S, K, T, r, sigma, seed=7)
    assert first == second


def test_different_seeds_give_different_answers():
    first, _ = monte_carlo_price(S, K, T, r, sigma, seed=1)
    second, _ = monte_carlo_price(S, K, T, r, sigma, seed=2)
    assert first != second


def test_error_shrinks_with_more_simulations():
    """100x more simulations should make the error about 10x smaller (square root rule)."""
    _, se_small = monte_carlo_price(S, K, T, r, sigma, n_sims=1_000, seed=42)
    _, se_large = monte_carlo_price(S, K, T, r, sigma, n_sims=100_000, seed=42)
    assert se_small / se_large == pytest.approx(10, rel=0.2)


def test_put_call_parity_approximately():
    """C - P should be close to S - K*e^(-rT), allowing for simulation noise."""
    c, _ = monte_carlo_price(S, K, T, r, sigma, seed=42)
    p, _ = monte_carlo_price(S, K, T, r, sigma, option_type="put", seed=42)
    assert c - p == pytest.approx(S - K * np.exp(-r * T), abs=0.2)