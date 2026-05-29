import numpy as np
import yfinance as yf
from scipy.stats import norm
from scipy.optimize import brentq
from hmmlearn.hmm import GaussianHMM

# --- 1. MARKET DATA --- #

ticker = 'MSFT'
data = yf.download(ticker, period="2y", auto_adjust=True)

closing_prices = data['Close']
daily_returns = closing_prices.pct_change().dropna()

std_deviation = daily_returns.std()
historical_volatility = std_deviation * np.sqrt(252)

S = float(closing_prices.iloc[-1])

print(f"Ticker: {ticker}")
print(f"Current Price (S): ${S:.2f}")
print(f"Historical Volatility: {float(historical_volatility.iloc[0]):.2%}")

# --- 2. BLACK-SCHOLES PRICER --- #

def black_scholes(S, K, T, r, sigma, option_type='call'):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))  # fixed
    d2 = d1 - sigma * np.sqrt(T)

    if option_type.lower() == 'call':
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type.lower() == 'put':
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    return price

# --- 3. IMPLIED VOLATILITY (IV) BACK-CALCULATION --- #

def implied_volatility(market_price, S, K, T, r, option_type='call'):
    def objective(sigma):  # fixed
        return black_scholes(S, K, T, r, sigma, option_type) - market_price

    try:
        iv = brentq(objective, 1e-6, 5.0)
        return iv
    except ValueError:
        return None
    
# --- EXTENSION --- HMM: MARKET REGIME DETECTION --- #

def fit_market_regime(returns, n_regimes=3):
    """
    Fits a Gaussian HMM to identify market regimes (e.g. low vol, high vol, crisis).
    n_regimes=3 by default — maps to the three regimes described in the README.
    """
    model = GaussianHMM(n_components=n_regimes, covariance_type="full", n_iter=100)
    # TODO: fit model and return regime labels
    pass