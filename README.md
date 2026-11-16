# Options Pricing & Volatility Surface Toolkit

A Python library for pricing options and building implied volatility surfaces from live market data.

> 🚧 Work in progress. Results, charts and a live demo link will go here.\

## Results

### Binomial tree converges to Black-Scholes

![Convergence chart](images/convergence.png)

Two independent pricing methods agree: as the number of steps grows, the binomial tree's price oscillates and settles onto the Black-Scholes value (€10.45), confirming both implementations.

### Monte Carlo estimate converges to Black-Scholes

![Monte Carlo chart](images/monte_carlo.png)

The Monte Carlo price and its 95% confidence interval narrow towards the Black-Scholes value as simulations increase. The error shrinks with the square root of the number of simulations: 100× more simulations gives a 10× tighter estimate.
Antithetic variates reduce the standard error by ~30% at no extra simulation cost (equivalent to doubling the number of simulations).

### Real market data: SPY volatility smile

![SPY volatility smile](images/smile_SPY_2026-12-18.png)

Implied volatilities calculated from live SPY option prices (out-of-the-money options, cleaned for wide spreads and stale quotes). If Black-Scholes were exactly right, this would be a flat line. Instead, implied volatility rises steeply for low strikes: deep out-of-the-money puts are priced at up to ~94% volatility against ~13% near the money, reflecting the market's demand for crash protection. A smaller upturn appears for far out-of-the-money calls.

## Features

- [x] Black-Scholes pricing for European calls and puts
- [x] Probability of finishing in the money
- [x] Analytic Greeks (delta, gamma, vega, theta, rho), validated against numerical derivatives
- [x] Binomial tree pricer (European and American)
- [x] Monte Carlo pricer
- [x] Implied volatility solver
- [ ] Volatility smiles and 3D surface from real option chains
- [ ] Delta-hedging simulation
- [ ] Interactive Streamlit dashboard

## Getting started

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m pytest
```

## Project structure

```
optionskit/      the library code
tests/           automated tests
notebooks/       exploration and charts
```
