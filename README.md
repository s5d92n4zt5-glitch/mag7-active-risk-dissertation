# Budgeting the Magnificent Seven

Python backtest of five US large-cap equity strategies against the S&P 500 (2015 to 2025), measuring how the rising weight of the Magnificent Seven affects relative risk, factor exposure, and performance. MSc dissertation, NEOMA Business School.

## What it does
Reconstructs the S&P 500 from point-in-time constituents, builds five strategies (cap-weighted, equal-weighted, capped at 5 percent, ex-Mag 7, and an active-fund proxy), and computes performance and relative-risk metrics, a Fama-French five-factor plus momentum attribution, a benchmark-dependency decomposition, and a Magnificent Seven budgeting frontier under a tracking-error constraint.

## Data availability
This repository contains code only. The input data are not included and must not be redistributed, because the CRSP datasets are obtained through WRDS under a proprietary licence. To reproduce the results, obtain the inputs and place them in a `06_data/` folder next to the code.
- S&P 500 constituents, the index series, and the mutual-fund data from WRDS (CRSP)
- Factor returns from the Kenneth French Data Library (public)

## How to run
    pip install -r requirements.txt
    python run_all.py

This runs the full pipeline in order and regenerates every table and figure used in the dissertation.

## Pipeline
- config.py, paths and parameters
- 01_load_data.py, load and clean data, build the monthly panel, validate the benchmark
- 02_strategies.py, construct the five strategies and their weights
- 03_metrics.py, performance and relative-risk metrics
- 04_factor_attribution.py, Fama-French five-factor plus momentum regressions
- 05_benchmark_dependence.py, Magnificent Seven contribution to return, drawdown, and tracking error
- 06_funds.py, active share and Magnificent Seven exposure of the twelve funds
- 07_tests_regression.py, Sharpe-difference tests and concentration regressions
- 08_concentration_subperiods.py, concentration metrics and sub-period performance
- 09_mag7_budget_frontier.py, the Magnificent Seven budgeting frontier
- run_all.py, runs the full pipeline in order
