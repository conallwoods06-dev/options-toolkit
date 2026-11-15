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

## Features

- [x] Black-Scholes pricing for European calls and puts
- [x] Probability of finishing in the money
- [x] Analytic Greeks (delta, gamma, vega, theta, rho), validated against numerical derivatives
- [x] Binomial tree pricer (European and American)
- [x] Monte Carlo pricer
- [ ] Implied volatility solver
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
