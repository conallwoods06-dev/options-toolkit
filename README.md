# Options Pricing & Volatility Surface Toolkit

A Python library for pricing options and building implied volatility surfaces from live market data.

> 🚧 Work in progress. Results, charts and a live demo link will go here.

## Features

- [x] Black-Scholes pricing for European calls and puts
- [x] Analytic Greeks (delta, gamma, vega, theta, rho), validated against numerical derivatives
- [ ] Binomial tree pricer (European and American)
- [ ] Monte Carlo pricer
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
