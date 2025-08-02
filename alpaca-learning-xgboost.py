import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import classification_report
from xgboost import XGBClassifier, plot_importance

# Helper indicator functions
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

# Load data
output_lines=[]
DATA_DIR='ticker-yearly'
TICKERS = ['LCID','TSLA','RIOT','INTC','SOFI','MARA','WBD','WMT','IBM','AMD','RBLX','NEE','META','AAPL','SPY','NVDA']
for ticker in TICKERS:
    file_path = os.path.join(DATA_DIR, f"{ticker}_yearly.csv")
    if not os.path.exists(file_path):
        print(f"{ticker}, was not found, skipping.")
        continue

    df = pd.read_csv(file_path, parse_dates=['Date'])
    df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].copy()

    # Feature engineering
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
    df['Target'] = (df['Close'].shift(-3) / df['Close'] - 1) > 0.01
    df['Target'] = df['Target'].astype(int)

    df.dropna(inplace=True)

    # Feature list
    features = ['SMA20', 'EMA20', 'RSI', 'MACD', 'ATR', 'OBV',
                'rolling_std', 'rolling_mean', 'Close/Open', 'High/Low',
                'Momentum', 'Volatility', 'VolumeChange', 'BB_Width'] + \
            [f'return_lag_{i}' for i in range(1, 6)]

    X = df[features]
    y = df['Target']

    X = X.replace([float('inf'),-float('inf')], pd.NA)
    X = X.dropna()

    y=y.loc[X.index]

    X=X.astype('float32')



    # Compute class imbalance
    pos_weight = (y == 0).sum() / (y == 1).sum()

    # XGBoost classifier with class balancing
    xgb = XGBClassifier(
        scale_pos_weight=pos_weight,
        eval_metric='logloss',
        random_state=42
    )

    # Cross-validation + hyperparameter tuning
    param_grid = {
        'max_depth': [3, 5],
        'learning_rate': [0.01, 0.1],
        'subsample': [0.8, 1],
        'colsample_bytree': [0.8, 1],
        'n_estimators': [100, 200]
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid = GridSearchCV(estimator=xgb, param_grid=param_grid, scoring='f1', cv=cv, verbose=1, n_jobs=-1)
    grid.fit(X, y)

    # Final model and evaluation
    best_model = grid.best_estimator_
    y_pred = best_model.predict(X)
    print(f"Classification report for {ticker}:\n")
    print(classification_report(y, y_pred))
    output_lines.append(f"Ticker: {ticker}\n{classification_report}")


out_path = "randomforest-report.txt"
with open(out_path,"w") as f:
    f.writelines(output_lines)

out_path


# Plot feature importance
plot_importance(best_model, max_num_features=15)
plt.title(f"{ticker} - XGBoost Feature Importance")
plt.tight_layout()
plt.show()


