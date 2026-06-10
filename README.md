# Options Mispricing Detection Using Machine Learning

## Overview
Can a machine learning model spot mispriced options before the market corrects? That's the question this project tries to answer.

The core idea: Black-Scholes gives a theoretical option price, but the market doesn't always agree. The gap between what the market implies and what our model expects is the mispricing signal.

## How It Works

1. **Market Data** — pulls 2 years of MSFT closing prices via `yfinance`, computes historical volatility
2. **HMM Regime Detection** — classifies each trading day into one of three market regimes (low vol, high vol, crisis) using a Gaussian Hidden Markov Model
3. **Synthetic Options** — generates ~18,000 option contracts with regime-dependent noise on top of Black-Scholes prices
4. **LSTM Model** — trained on 10-day IV surface sequences to predict next-day ATM implied volatility
5. **Mispricing Signal** — flags options where market IV deviates >5% from the model's prediction as overpriced or underpriced
6. **Backtesting** — simulates trades on flagged signals, reports P&L, win rate, Sharpe ratio, and max drawdown

## Data Sources
- **Prices:** `yfinance` (MSFT, 2-year window)
- **Options:** Synthetically generated from price history
- **Planned:** CBOE or OptionMetrics for real IV surface validation

## Dependencies
```bash
uv pip install yfinance hmmlearn scikit-learn torch pandas numpy scipy --system
```