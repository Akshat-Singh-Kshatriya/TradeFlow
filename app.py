import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

# ==========================================
# 1. ADVANCED CONFIGURATION & DATA SIMULATION
# ==========================================
class MarketConfig:
    SPOT_PRICE = 6986.00
    ATM_STRIKE = 7000.00
    RISK_FREE_RATE = 0.054  
    DAYS_TO_EXPIRY = 23
    VOLATILITY = 0.25       # 25% IV
    SIM_DAYS = 365 * 2      # Generate 2 years of data

np.random.seed(42)

def generate_realistic_data(config):
    dt = 1/252
    prices = [6000]
    vol = config.VOLATILITY
    
    for _ in range(config.SIM_DAYS):
        # Stochastic volatility component
        shock = np.random.normal(0, 1)
        vol = vol + 0.1 * (0.25 - vol) + 0.05 * shock # Mean reverting vol
        vol = max(0.1, vol) # Floor volatility
        
        # Geometric Brownian Motion
        drift = (config.RISK_FREE_RATE - 0.5 * vol**2) * dt
        diffusion = vol * np.sqrt(dt) * np.random.normal(0, 1)
        price = prices[-1] * np.exp(drift + diffusion)
        prices.append(price)
        
    dates = pd.date_range(end=pd.Timestamp.today(), periods=len(prices))
    df = pd.DataFrame({'Date': dates, 'Close': prices})
    return df

# ==========================================
# 2. FEATURE ENGINEERING (TECHNICAL INDICATORS)
# ==========================================
def add_technical_indicators(df):
    df = df.copy()
    
    # 1. RSI (Relative Strength Index)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # 2. MACD
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    
    # 3. Bollinger Bands
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['BB_Upper'] = df['MA20'] + (df['Close'].rolling(window=20).std() * 2)
    df['BB_Lower'] = df['MA20'] - (df['Close'].rolling(window=20).std() * 2)
    
    # 4. Historical Volatility
    df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
    df['Hist_Vol'] = df['Log_Ret'].rolling(window=21).std() * np.sqrt(252)
    
    return df.dropna()

# Generate and prep data
raw_data = generate_realistic_data(MarketConfig)
ml_data = add_technical_indicators(raw_data)

# ==========================================
# 3. ADVANCED MACHINE LEARNING (Gradient Boosting)
# ==========================================
print("--- TRAINING ADVANCED ML MODEL ---")

# Target: Future Price
ml_data['Target'] = ml_data['Close'].shift(-MarketConfig.DAYS_TO_EXPIRY)
model_data = ml_data.dropna()

features = ['Close', 'RSI', 'MACD', 'Hist_Vol', 'BB_Upper', 'BB_Lower']
X = model_data[features]
y = model_data['Target']

# Time Series Split (Prevent data leakage)
tscv = TimeSeriesSplit(n_splits=5)
model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42)

for train_index, test_index in tscv.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    model.fit(X_train, y_train)

# Current Prediction
current_features = X.iloc[[-1]]
predicted_price = model.predict(current_features)[0]
current_spot = ml_data.iloc[-1]['Close']

print(f"Current Spot: ₹{current_spot:.2f}")
print(f"ML Predicted Spot (in {MarketConfig.DAYS_TO_EXPIRY} days): ₹{predicted_price:.2f}")

# ==========================================
# 4. BLACK-SCHOLES PRICING & GREEKS
# ==========================================
def black_scholes(S, K, T, r, sigma, option_type='call'):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == 'call':
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        delta = norm.cdf(d1)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        delta = norm.cdf(d1) - 1
        
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * norm.pdf(d1) * np.sqrt(T) / 100 # Divided by 100 for percentage change
    theta = (- (S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
    
    return price, delta, gamma, theta, vega

# Strategy Setup: Iron Butterfly (Short ATM, Long Wings)
# Strikes
K_ATM = round(current_spot / 100) * 100
K_LOWER = K_ATM - 200
K_UPPER = K_ATM + 200

T_years = MarketConfig.DAYS_TO_EXPIRY / 365

# Pricing
p_lower, d_l, g_l, t_l, v_l = black_scholes(current_spot, K_LOWER, T_years, MarketConfig.RISK_FREE_RATE, MarketConfig.VOLATILITY)
p_atm, d_a, g_a, t_a, v_a = black_scholes(current_spot, K_ATM, T_years, MarketConfig.RISK_FREE_RATE, MarketConfig.VOLATILITY)
p_upper, d_u, g_u, t_u, v_u = black_scholes(current_spot, K_UPPER, T_years, MarketConfig.RISK_FREE_RATE, MarketConfig.VOLATILITY)

skew_adj = 1.02 # 2% higher IV for wings
p_lower_skewed = p_lower * skew_adj
p_upper_skewed = p_upper * skew_adj

net_debit = p_lower_skewed + p_upper_skewed - (2 * p_atm)

print(f"\n[ADVANCED STRATEGY: LONG CALL BUTTERFLY]")
print(f"Leg 1: Buy Call {K_LOWER} @ {p_lower_skewed:.2f} (Delta: {d_l:.2f})")
print(f"Leg 2: Sell 2 Calls {K_ATM} @ {p_atm:.2f} (Delta: {d_a:.2f})")
print(f"Leg 3: Buy Call {K_UPPER} @ {p_upper_skewed:.2f} (Delta: {d_u:.2f})")
print(f"Net Debit: ₹{net_debit:.2f}")

# Net Portfolio Greeks
net_delta = d_l + d_u - (2 * d_a)
net_gamma = g_l + g_u - (2 * g_a)
net_theta = t_l + t_u - (2 * t_a)
net_vega = v_l + v_u - (2 * v_a)

print(f"\n[PORTFOLIO GREEKS]")
print(f"Delta: {net_delta:.4f} (Directional Risk)")
print(f"Gamma: {net_gamma:.4f} (Curvature Risk)")
print(f"Theta: {net_theta:.2f} (Daily Time Decay Gain)")
print(f"Vega:  {net_vega:.2f} (Sensitivity to Volatility)")

# ==========================================
# 5. MONTE CARLO SIMULATION (PROBABILITY OF PROFIT)
# ==========================================
SIMULATIONS = 10000
simulated_end_prices = []

for i in range(SIMULATIONS):
    # Random walk to expiry
    drift = (MarketConfig.RISK_FREE_RATE - 0.5 * MarketConfig.VOLATILITY**2) * T_years
    diffusion = MarketConfig.VOLATILITY * np.sqrt(T_years) * np.random.normal()
    price_t = current_spot * np.exp(drift + diffusion)
    simulated_end_prices.append(price_t)

simulated_end_prices = np.array(simulated_end_prices)

# Calculate P/L for every simulation
payoffs = np.maximum(simulated_end_prices - K_LOWER, 0) - \
          2 * np.maximum(simulated_end_prices - K_ATM, 0) + \
          np.maximum(simulated_end_prices - K_UPPER, 0)
net_pnls = payoffs - net_debit

pop = np.mean(net_pnls > 0) * 100 # Probability of Profit
expected_value = np.mean(net_pnls)
var_95 = np.percentile(net_pnls, 5)

print(f"\n[RISK ANALYSIS]")
print(f"Probability of Profit (PoP): {pop:.2f}%")
print(f"Expected Value per Share: ₹{expected_value:.2f}")
print(f"Value at Risk (95%): ₹{abs(var_95):.2f}")


