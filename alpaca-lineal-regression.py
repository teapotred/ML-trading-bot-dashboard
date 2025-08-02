import pandas as pd
import os
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pandas_ta as ta

DATA_DIR = 'Tickers'
TICKERS = ['LCID','TSLA','RIOT','INTC','SOFI','MARA','WBD','WMT','IBM','AMD','RBLX','NEE','META','AAPL','SPY','NVDA']

all_results = {}

for ticker in TICKERS:
    file_path = os.path.join(DATA_DIR, f"{ticker}_yearly.csv")
    if not os.path.exists(file_path):
        print(f"File for {ticker} not found. Skipping.")
        continue

    df = pd.read_csv(file_path)
    df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].copy()
    df.dropna(inplace=True)

    # === Manual Technical Indicators ===
    df['Return'] = df['Close'].pct_change()
    df['Volatility'] = df['Return'].rolling(window=14).std()
    df['Momentum'] = df['Close'] - df['Close'].shift(10)
    df['LogVolume'] = (df['Volume'].replace(0, 1)).apply(lambda x: pd.np.log(x))

    # Target: 3-day future return
    df['Target'] = df['Close'].shift(-3).pct_change(periods=3)

    df.dropna(inplace=True)

    features = ['Return', 'Volatility', 'Momentum', 'LogVolume']
    X = df[features]
    y = df['Target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False)

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"\nTicker: {ticker}")
    print(f"Mean Squared Error: {mse:.6f}")
    print(f"R^2 Score: {r2:.4f}")   

    # Save results
    all_results[ticker] = {
        'mse': mse,
        'r2': r2,
        'predictions': y_pred[:5],
        'actuals': y_test.values[:5]
    }
