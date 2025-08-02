import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import classification_report
from xgboost import XGBClassifier, plot_importance

# === CONFIG ===
DATA_DIR = 'ticker-yearly'
TICKERS = ["RIOT","TSLA","SPY","LCID","SOFI",]  # Change this manually as needed
MODEL_DIR = 'xgb_models'
os.makedirs(MODEL_DIR, exist_ok=True)  # Create folder if it doesn't exist

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

    # Bollinger Band Width
    rolling_mean = df['Close'].rolling(window=20).mean()
    rolling_std = df['Close'].rolling(window=20).std()
    df['BB_Width'] = (rolling_mean + 2*rolling_std) - (rolling_mean - 2*rolling_std)

    # Target label: price goes up >1% in 3 days
    # Another Target label I'll try is to check if it goes up in the next 5 days
    #future_return = (df['Close'].shift(-5) + df['Close'].shift(-6) + df['Close'].shift(-7)) / 3 / df['Close'] - 1
    future_return = (df['Close'].shift(-5) / df['Close'] - 1)
    df['Target'] = (future_return>0.001).astype(int)


    #df['Target'] = (df['Close'].shift(-7) / df['Close'] - 1) > 0.01
    #df['Target'] = df['Target'].astype(int)

    df.dropna(inplace=True)

    features = ['SMA20', 'EMA20', 'RSI', 'MACD', 'ATR', 'OBV',
                'rolling_std', 'rolling_mean', 'Close/Open', 'High/Low',
                'Momentum', 'Volatility', 'VolumeChange', 'BB_Width'] + \
               [f'return_lag_{i}' for i in range(1, 6)]

    X = df[features].replace([float('inf'), -float('inf')], np.nan).dropna().astype('float32')
    y = df['Target'].loc[X.index]

    # === Time-based split for future-aware testing ===
    split_idx = int(len(X) * 0.8)  # 80% for training, 20% for future testing
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    pos_weight = (y == 0).sum() / (y == 1).sum()

    xgb = XGBClassifier(
    max_depth=3,             # limit how deep trees go (less complex)
    learning_rate=0.05,      # slower learning
    subsample=0.8,           # don't use all rows for each tree
    colsample_bytree=0.8,    # don't use all features
    reg_alpha=0.5,           # L1 regularization (lasso)
    reg_lambda=1.0,          # L2 regularization (ridge)
    n_estimators=100,
    scale_pos_weight=pos_weight,
    eval_metric='logloss',
    random_state=42
    )

    param_grid = {
        'max_depth': [3, 5],
        'learning_rate': [0.01, 0.1],
        'subsample': [0.8, 1],
        'colsample_bytree': [0.8, 1],
        'n_estimators': [100, 200]
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid = GridSearchCV(estimator=xgb, param_grid=param_grid, scoring='f1', cv=cv, verbose=1, n_jobs=-1)
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)
    print(f"\nClassification report for {ticker}:\n")
    print(classification_report(y_test, y_pred))
    print(f"{ticker} | Ups: {(df['Target'] == 1).sum()} | Downs: {(df['Target'] == 0).sum()}")


    # === Save the model ===
    model_path = os.path.join(MODEL_DIR, f"{ticker}_model_v1.json")
    best_model.save_model(model_path)
    print(f"✅ Model saved to {model_path}")


