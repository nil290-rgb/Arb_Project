Options Mispricing Detection Using Machine Learning

Project Overview

This project explores whether a machine learning model can learn from historical options data to identify arbitrage opportunities before market correction (price convergence), and whether acting on those signals can be profitable after accounting for transaction costs.
The core idea: Black-Scholes gives a theoretical option price, but the market doesn't always agree. The gap between what the market implies and what our model expects is the mispricing signal.

How It Works
Step 1 — Theoretical Benchmark (vol_mispricing.py)

Pulls historical closing prices for a stock (currently MSFT) using yfinance
Computes historical volatility from daily returns
Prices options using the Black-Scholes model given market inputs (spot price, strike, time to expiry, risk-free rate)
Backs out Implied Volatility (IV) from the observed market price — i.e. what volatility the market is assuming
Compares IV against historical volatility to flag potential mispricings:

IV significantly higher than historical vol → market is overpricing → sell signal
IV significantly lower than historical vol → market is underpricing → buy signal



Step 2 — ML Model (in progress)

Train a machine learning model on historical options data to predict what IV should be given current market conditions
The model learns patterns that Black-Scholes ignores — volatility skew, term structure, VIX level, and regime-dependent behavior
The mispricing signal will then compare current market IV against the ML model's prediction rather than raw historical vol — a much stronger and context-aware signal


Planned Extension — Hidden Markov Model (HMM)
Future iterations will incorporate a Hidden Markov Model to identify market regimes (low volatility, high volatility, crisis). Instead of using a single flat historical volatility estimate, the model will dynamically adjust its volatility expectations based on the current regime — making the mispricing signal more robust across different market conditions.
Initial skeleton implemented in vol_mispricing.py.

------ Data Sources -------

Stock price history: yfinance (currently MSFT)
Options data for ML training (Step 2): to be sourced from CBOE historical data, OptionMetrics, or synthetic chains derived from equity price histories
