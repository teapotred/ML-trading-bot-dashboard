import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

DATA_DIR = "ticker-yearly"
TICKERS = ['LCID','TSLA','RIOT','INTC','SOFI','MARA','WBD','WMT','IBM','AMD','RBLX','NEE']

all_results = {}

# Manual RSI calculation
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# Manual MACD calculation
def compute_macd(series, fast=12, slow=26):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    return ema_fast - ema_slow

# Manual ATR calculation
def compute_atr(high, low, close, period=14):
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()

# Manual OBV calculation
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

output_lines = []

for ticker in TICKERS:
    file_path = os.path.join(DATA_DIR, f"{ticker}_yearly.csv")
    if not os.path.exists(file_path):
        print(f"File for {ticker} not found. Skipping.")
        continue

    df = pd.read_csv(file_path, parse_dates=['Date'])
    df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].copy()

    # Compute technical indicators manually
    df['SMA20'] = df['Close'].rolling(window=20).mean()
    df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['RSI'] = compute_rsi(df['Close'])
    df['MACD'] = compute_macd(df['Close'])
    df['ATR'] = compute_atr(df['High'], df['Low'], df['Close'])
    df['OBV'] = compute_obv(df['Close'], df['Volume'])

    # Lagged returns and rolling stats
    for i in range(1, 6):
        df[f'return_lag_{i}'] = df['Close'].pct_change(i).shift(1)

    df['rolling_std'] = df['Close'].rolling(window=5).std()
    df['rolling_mean'] = df['Close'].rolling(window=5).mean()

    # Target: Price will be higher 3 days from now
    df['Target'] = (df['Close'].shift(-3) > df['Close']).astype(int)
    df.dropna(inplace=True)

    # Features
    features = ['SMA20', 'EMA20', 'RSI', 'MACD', 'ATR', 'OBV',
                'rolling_std', 'rolling_mean'] + [f'return_lag_{i}' for i in range(1, 6)]
    X = df[features]
    y = df['Target']

    # Time-aware train-test split
    split_index = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    report = classification_report(y_test, y_pred, output_dict=False)
    output_lines.append(f"Ticker: {ticker}\n{report}\n")

    print(f"\nTicker: {ticker}")
    print(classification_report(y_test, y_pred))

    # Store for later use
    all_results[ticker] = {
        'model': model,
        'report': classification_report(y_test, y_pred, output_dict=True)
    }

out_path = "randomforest-report.txt"
with open(out_path,"w") as f:
    f.writelines(output_lines)

out_path


