import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from xgboost import XGBRegressor

# === CONFIG ===
DATA_DIR = 'ticker-yearly'
TICKERS = ['DNN', 'LCID', 'RIOT', 'SOFI', 'SPY', 'TSLA']
MODEL_DIR = 'xgb_models_reg'
os.makedirs(MODEL_DIR, exist_ok=True)

# === Helper Indicator Functions ===
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def compute_macd(series, fast=12, slow=26):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    return ema_fast - ema_slow

def compute_atr(high, low, close, period=14):
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()

def compute_obv(close, volume):
    obv = [0]
    for i in range(1, len(close)):
        if close[i] > close[i - 1]:
            obv.append(obv[-1] + volume[i])
        elif close[i] < close[i - 1]:
            obv.append(obv[-1] - volume[i])
        else:
            obv.append(obv[-1])
    return pd.Series(obv, index=close.index)

# === Main Training Loop ===
results = []
for ticker in TICKERS:
    file_path = os.path.join(DATA_DIR, f"{ticker}_yearly.csv")
    if not os.path.exists(file_path):
        print(f"{ticker} not found. Skipping.")
        continue

    df = pd.read_csv(file_path, parse_dates=['Date'])
    df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].copy()

    # === Feature Engineering ===
    df['SMA20'] = df['Close'].rolling(window=20).mean()
    df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['RSI'] = compute_rsi(df['Close'])
    df['MACD'] = compute_macd(df['Close'])
    df['ATR'] = compute_atr(df['High'], df['Low'], df['Close'])
    df['OBV'] = compute_obv(df['Close'], df['Volume'])

    for i in range(1, 6):
        df[f'return_lag_{i}'] = df['Close'].pct_change(i).shift(1)

    df['rolling_std'] = df['Close'].rolling(window=5).std()
    df['rolling_mean'] = df['Close'].rolling(window=5).mean()
    df['Close/Open'] = df['Close'] / df['Open']
    df['High/Low'] = df['High'] / df['Low']
    df['Momentum'] = df['Close'] - df['Close'].shift(3)
    df['Volatility'] = df['Close'].rolling(window=5).std()
    df['VolumeChange'] = df['Volume'].pct_change()
    rolling_mean = df['Close'].rolling(window=20).mean()
    rolling_std = df['Close'].rolling(window=20).std()
    df['BB_Width'] = (rolling_mean + 2 * rolling_std) - (rolling_mean - 2 * rolling_std)

    # === Regression Target: 7-day forward return ===
    df['Target'] = df['Close'].shift(-7) / df['Close'] - 1
    df.dropna(inplace=True)

    features = ['SMA20', 'EMA20', 'RSI', 'MACD', 'ATR', 'OBV',
                'rolling_std', 'rolling_mean', 'Close/Open', 'High/Low',
                'Momentum', 'Volatility', 'VolumeChange', 'BB_Width'] + \
               [f'return_lag_{i}' for i in range(1, 6)]

    X = df[features].replace([float('inf'), -float('inf')], np.nan).dropna().astype('float32')
    y = df['Target'].loc[X.index]

    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    xgb = XGBRegressor(
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.5,
        reg_lambda=1.0,
        n_estimators=200,
        random_state=42
    )

    xgb.fit(X_train, y_train)
    y_pred = xgb.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    results.append((ticker, mae, mse, r2))

    model_path = os.path.join(MODEL_DIR, f"{ticker}_reg_model.json")
    xgb.save_model(model_path)
    print(f"✅ Saved regressor model for {ticker} to {model_path}")

