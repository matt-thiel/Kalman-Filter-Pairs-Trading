# Kalman-Filter-Pairs-Trading
Pairs trading using the Kalman filter for time-varying parameters, inspired by "Pairs Trading with R" by Daniel Palomar.

## Project Description
- kalman_pairs_case_study.ipynb first reproduces the results of the original paper by Palomar, then tests the strategy out-of-sample (OOS) using proper training windows. The strategy outperforms the equal-weighted benchmark with a Sharpe Ratio of approximately 0.7. 
- kalman_pairs_iterative.py implements the strategy iteratively, allowing for use with common backtesting/live trading libraries. 
- kalman_pairs_dynamic.py implements the strategy with dynamic pairs selection, comparing the cointegration and distance approach to a benchmark.
