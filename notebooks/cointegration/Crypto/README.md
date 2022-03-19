# Crypto
Warren Fulton, John Palmer, Jamie Voros

# Analysis of patterns in major Crypto-currencies

## Data Acquisition
The file 'findCointedPairs.ipynb' finds all sufficiently cointegrated pairs of the available currencies.

It takes a list of all 78 cryptocurrencies tickers that we found data for and then finds all possible pairs between those assets. Then these pairs are passed into the cointegration screening process. Here we run the Statsmodels function for cointegration, 'coint()', on the available historical data to determine the associated p-value of the price distrbutions of the two assets, and if the p-value is less than a 5% significance level, then it is added to the list of cointegrated pairs.

## Backesting
We backtested a series of pairs trades.

Backtesting was written in a Jupyter notebook available in this repo (BackTesting.ipynb).

The backtest implements the pairs trading algorithm with window function and runs on existing data from each asset pair. The backtest runs between dates for which both assets have data. We are using simply the closing price for each trading day (days with no trading are skipped).
