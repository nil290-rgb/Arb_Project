import numpy as np
import yfinance as yf
from scipy.stats import norm
from scipy.optimize import brentq
from hmmlearn.hmm import GaussianHMM
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

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
    model = GaussianHMM(n_components=n_regimes, covariance_type="full", n_iter=100,
        random_state=42
    )
    model.fit(returns.reshape(-1, 1))                                
    regime_labels = model.predict(returns.reshape(-1, 1))                    
 
    # Sort regimes by variance so labels are consistent:
    # 0 = lowest vol, 1 = mid vol, 2 = highest vol (crisis)
    variances = [np.var(returns[regime_labels == i]) for i in range(n_regimes)]
    sorted_regimes = np.argsort(variances)
    regime_map = {old: new for new, old in enumerate(sorted_regimes)}
    regime_labels = np.array([regime_map[r] for r in regime_labels])
 
    return regime_labels, model
 
 
print("\nFitting HMM for regime detection...")
returns_clean = daily_returns.values.flatten()
regime_labels, hmm_model = fit_market_regime(returns_clean)
 
regime_series = pd.Series(regime_labels, index=daily_returns.index, name='regime')
regime_counts = pd.Series(regime_labels).value_counts().sort_index()
regime_names  = {0: 'Low Vol', 1: 'High Vol', 2: 'Crisis'}
print("Regime distribution:")
for r, count in regime_counts.items():
    print(f"  Regime {r} ({regime_names[r]}): {count} days")
 
 
# ── 5. SYNTHETIC OPTIONS DATA GENERATION ─────────────────────────────────────
 
def generate_synthetic_options(closing_prices, regime_series, r=0.05):
    """
    For each trading day, generate synthetic options at multiple strikes and expiries.
    Uses Black-Scholes as the base pricer, adds realistic noise to simulate market deviations.
    """
    records = []
 
    moneyness_levels = [0.90, 0.95, 1.00, 1.05, 1.10]  # % of spot price
    expiries_years   = [1/12, 3/12, 6/12, 1.0]          # 1m, 3m, 6m, 1y
    option_types     = ['call', 'put']
 
    # Rolling 30-day historical vol for each day
    hist_vol_series = closing_prices.pct_change().rolling(30).std() * np.sqrt(252)
 
    dates = closing_prices.index[30:]  # skip first 30 days (rolling window warmup)
 
    for date in dates:
        spot      = float(closing_prices.loc[date].item())
        hist_vol  = float(hist_vol_series.loc[date].item())
        regime    = int(regime_series.loc[date]) if date in regime_series.index else 0
 
        if np.isnan(hist_vol) or hist_vol <= 0:
            continue
 
        for m in moneyness_levels:
            K = spot * m
            for T in expiries_years:
                for opt_type in option_types:
                    bs_price = black_scholes(spot, K, T, r, hist_vol, opt_type)
 
                    # Add market noise — regime-dependent
                    # Crisis regime has more noise (market deviates more from theory)
                    noise_scale = {0: 0.02, 1: 0.05, 2: 0.10}[regime]
                    noise = np.random.normal(0, noise_scale * bs_price)
                    market_price = max(bs_price + noise, 1e-4)
 
                    iv = implied_volatility(market_price, spot, K, T, r, opt_type)
                    if iv is None:
                        continue
 
                    records.append({
                        'date':        date,
                        'spot':        spot,
                        'strike':      K,
                        'moneyness':   spot / K,
                        'T':           T,
                        'hist_vol':    hist_vol,
                        'regime':      regime,
                        'option_type': 0 if opt_type == 'call' else 1,
                        'bs_price':    bs_price,
                        'market_price':market_price,
                        'iv':          iv,
                    })
 
    return pd.DataFrame(records)
 
 
print("\nGenerating synthetic options data...")
options_df = generate_synthetic_options(closing_prices, regime_series)
print(f"Generated {len(options_df)} synthetic options")
print(options_df.head())
 
 
# ── 6. BUILD IV SURFACE SNAPSHOTS (PIVOT A) ──────────────────────────────────
# Instead of individual contract rows, each day becomes one snapshot:
# ATM_IV, OTM_Call_IV, OTM_Put_IV, Hist_Vol, Regime
# This gives the LSTM a clean time series where day t-1 flows into day t.
 
def build_iv_surface(df):
    """
    For each date, compute:
      - ATM_IV:      IV of options closest to moneyness = 1.0
      - OTM_Call_IV: IV of calls with moneyness < 1.0 (OTM calls)
      - OTM_Put_IV:  IV of puts with moneyness > 1.0 (OTM puts)
      - hist_vol:    historical volatility for that day
      - regime:      HMM regime label
    """
    records = []
    for date, group in df.groupby('date'):
        calls = group[group['option_type'] == 0]
        puts  = group[group['option_type'] == 1]
 
        # ATM: closest moneyness to 1.0
        atm_row = group.iloc[(group['moneyness'] - 1.0).abs().argsort()].iloc[0]
        atm_iv  = atm_row['iv']
 
        # OTM call: calls where stock > strike (moneyness > 1)
        otm_calls = calls[calls['moneyness'] > 1.0]
        otm_call_iv = otm_calls['iv'].mean() if len(otm_calls) > 0 else atm_iv
 
        # OTM put: puts where stock < strike (moneyness < 1)
        otm_puts = puts[puts['moneyness'] < 1.0]
        otm_put_iv = otm_puts['iv'].mean() if len(otm_puts) > 0 else atm_iv
 
        records.append({
            'date':         date,
            'atm_iv':       atm_iv,
            'otm_call_iv':  otm_call_iv,
            'otm_put_iv':   otm_put_iv,
            'hist_vol':     group['hist_vol'].iloc[0],
            'regime':       group['regime'].iloc[0],
        })
 
    return pd.DataFrame(records).sort_values('date').reset_index(drop=True)
 
 
print("\nBuilding IV surface snapshots...")
surface_df = build_iv_surface(options_df)
print(f"Surface snapshots: {len(surface_df)} days")
print(surface_df.head())
 
# ── 6b. CHRONOLOGICAL TRAIN/TEST SPLIT (PIVOT B) ─────────────────────────────
 
unique_dates = surface_df['date'].unique()
split_idx    = int(len(unique_dates) * 0.8)
train_dates  = unique_dates[:split_idx]
test_dates_arr = unique_dates[split_idx:]
 
train_surface = surface_df[surface_df['date'].isin(train_dates)].reset_index(drop=True)
test_surface  = surface_df[surface_df['date'].isin(test_dates_arr)].reset_index(drop=True)
 
# ── 6c. SCALE AND CREATE SEQUENCES ────────────────────────────────────────────
 
SEQUENCE_LENGTH = 10
feature_cols    = ['atm_iv', 'otm_call_iv', 'otm_put_iv', 'hist_vol', 'regime']
target_col      = 'atm_iv'  # predict tomorrow's ATM IV
 
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
 
train_X = scaler_X.fit_transform(train_surface[feature_cols].values)
train_y = scaler_y.fit_transform(train_surface[target_col].values.reshape(-1, 1))
test_X  = scaler_X.transform(test_surface[feature_cols].values)
test_y  = scaler_y.transform(test_surface[target_col].values.reshape(-1, 1))
 
def create_sequences(X, y, seq_len):
    Xs, ys = [], []
    for i in range(len(X) - seq_len):
        Xs.append(X[i:i + seq_len])
        ys.append(y[i + seq_len])
    return np.array(Xs), np.array(ys)
 
X_train, y_train = create_sequences(train_X, train_y, SEQUENCE_LENGTH)
X_test,  y_test  = create_sequences(test_X,  test_y,  SEQUENCE_LENGTH)
test_dates       = test_surface['date'].values[SEQUENCE_LENGTH:]
 
print(f"Train sequences: {len(X_train)} | Test sequences: {len(X_test)}")
 
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32)
X_test_t  = torch.tensor(X_test,  dtype=torch.float32)
y_test_t  = torch.tensor(y_test,  dtype=torch.float32)
 
train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=32, shuffle=False)
 
 
# ── 7. LSTM MODEL ─────────────────────────────────────────────────────────────
 
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, output_size=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc   = nn.Linear(hidden_size, output_size)
 
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out
 
 
model     = LSTMModel(input_size=len(feature_cols))
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
 
 
# ── 8. TRAINING ───────────────────────────────────────────────────────────────
 
EPOCHS = 50
 
print("\nTraining LSTM...")
for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        preds = model(X_batch)
        loss  = criterion(preds, y_batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
 
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{EPOCHS} — Loss: {epoch_loss/len(train_loader):.6f}")
 
 
# ── 9. EVALUATION ─────────────────────────────────────────────────────────────
 
model.eval()
with torch.no_grad():
    y_pred_scaled = model(X_test_t).numpy()
 
y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_true = scaler_y.inverse_transform(y_test)
 
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
print(f"\nTest RMSE (IV prediction): {rmse:.6f}")
 
 
# ── 10. MISPRICING SIGNAL ─────────────────────────────────────────────────────
 
THRESHOLD = 0.05  # 5% IV gap to flag as mispricing
 
results = pd.DataFrame({
    'actual_iv':    y_true.flatten(),
    'predicted_iv': y_pred.flatten(),
})
 
results['iv_gap'] = results['actual_iv'] - results['predicted_iv']
results['signal'] = results['iv_gap'].apply(
    lambda gap: 'OVERPRICED — consider selling' if gap >  THRESHOLD else
                'UNDERPRICED — consider buying' if gap < -THRESHOLD else
                'FAIRLY PRICED'
)
 
print("\n── Mispricing Signals (first 20 rows) ──────────────────")
print(results.head(20).to_string(index=False))
 
overpriced  = (results['signal'] == 'OVERPRICED — consider selling').sum()
underpriced = (results['signal'] == 'UNDERPRICED — consider buying').sum()
fair        = (results['signal'] == 'FAIRLY PRICED').sum()
 
print(f"\nOverpriced:    {overpriced}")
print(f"Underpriced:   {underpriced}")
print(f"Fairly priced: {fair}")
 
 
# ── 11. BACKTESTING ───────────────────────────────────────────────────────────
"""
Simple backtesting logic:
- OVERPRICED signal → sell the option, collect premium, buy back at BS theoretical price
- UNDERPRICED signal → buy the option, sell when it converges to BS theoretical price
- P&L = difference between market price and BS theoretical price
- Transaction cost: 0.5% of option price per trade (buy + sell = 1% total)
"""
 
TRANSACTION_COST = 0.01  # 1% round-trip
 
# Merge results with test_surface using dates for proper temporal alignment
results['date'] = pd.Series(test_dates).values
backtest = results.merge(
    test_surface[['date', 'hist_vol']],
    on='date', how='left'
).reset_index(drop=True)
# Approximate market/bs price from IV gap for P&L (surface-level backtesting)
backtest['market_price'] = backtest['actual_iv'] * 100   # proxy: IV * 100 as price unit
backtest['bs_price']     = backtest['predicted_iv'] * 100
 
def compute_pnl(row):
    signal      = row['signal']
    market_price = row['market_price']
    bs_price     = row['bs_price']
    txn_cost     = TRANSACTION_COST * market_price
 
    if signal == 'OVERPRICED — consider selling':
        # Sell at market price, buy back at BS theoretical price
        pnl = (market_price - bs_price) - txn_cost
    elif signal == 'UNDERPRICED — consider buying':
        # Buy at market price, sell at BS theoretical price
        pnl = (bs_price - market_price) - txn_cost
    else:
        pnl = 0.0
 
    return pnl
 
backtest['pnl'] = backtest.apply(compute_pnl, axis=1)
 
# Summary stats
total_trades    = (backtest['signal'] != 'FAIRLY PRICED').sum()
profitable      = (backtest['pnl'] > 0).sum()
total_pnl       = backtest['pnl'].sum()
avg_pnl         = backtest.loc[backtest['signal'] != 'FAIRLY PRICED', 'pnl'].mean()
win_rate        = profitable / total_trades * 100 if total_trades > 0 else 0
 
# Cumulative P&L over time
backtest['cumulative_pnl'] = backtest['pnl'].cumsum()
 
# Sharpe ratio (annualised, assuming 252 trading days)
daily_pnl   = backtest.groupby('date')['pnl'].sum()
sharpe      = (daily_pnl.mean() / daily_pnl.std()) * np.sqrt(252) if daily_pnl.std() > 0 else 0
 
# Max drawdown
cumulative  = backtest['cumulative_pnl']
rolling_max = cumulative.cummax()
drawdown    = cumulative - rolling_max
max_drawdown = drawdown.min()
 
print("\n── Backtesting Results ──────────────────────────────────")
print(f"Total trades:       {total_trades}")
print(f"Profitable trades:  {profitable}")
print(f"Win rate:           {win_rate:.1f}%")
print(f"Total P&L:          ${total_pnl:.4f}")
print(f"Average P&L/trade:  ${avg_pnl:.4f}")
print(f"Sharpe Ratio:       {sharpe:.2f}")
print(f"Max Drawdown:       ${max_drawdown:.4f}")
print(f"\nFinal Cumulative P&L: ${backtest['cumulative_pnl'].iloc[-1]:.4f}")
