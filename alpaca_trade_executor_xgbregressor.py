import os
import pandas as pd
import xgboost as xgb
import numpy as np

# === CONFIG ===
DATA_DIR = 'ticker-yearly'
MODEL_DIR = 'xgb_models_reg'
TICKERS = ['DNN', 'LCID', 'RIOT', 'SOFI', 'SPY', 'TSLA']

# === Features used during training ===
FEATURES = ['SMA20', 'EMA20', 'RSI', 'MACD', 'ATR', 'OBV',
            'rolling_std', 'rolling_mean', 'Close/Open', 'High/Low',
            'Momentum', 'Volatility', 'VolumeChange', 'BB_Width'] + \
           [f'return_lag_{i}' for i in range(1, 6)]

for ticker in TICKERS:
    print(f"\n🔍 Predicting return for {ticker}...")

    try:
        file_path = os.path.join(DATA_DIR, f"{ticker}_yearly.csv")
        if not os.path.exists(file_path):
            print(f"❌ Data for {ticker} not found.")
            continue

        df = pd.read_csv(file_path)

        # Same feature engineering as training
        df['SMA20'] = df['Close'].rolling(window=20).mean()
        df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
        df['RSI'] = df['Close'].pct_change().rolling(14).apply(lambda x: (x[x > 0].sum() / abs(x).sum()) * 100)
        df['MACD'] = df['Close'].ewm(span=12).mean() - df['Close'].ewm(span=26).mean()
        df['ATR'] = (df['High'] - df['Low']).rolling(14).mean()
        df['OBV'] = (df['Volume'] * (df['Close'].diff() > 0).apply(lambda x: 1 if x else -1)).cumsum()

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

        df.dropna(inplace=True)
        latest_features = df[FEATURES].iloc[-1:].astype('float32')

        # Load model
        model_path = os.path.join(MODEL_DIR, f"{ticker}_reg_model.json")
        model = xgb.XGBRegressor()
        model.load_model(model_path)

        # Predict
        predicted_return = model.predict(latest_features)[0]
        predicted_pct = round(predicted_return * 100, 2)

        print(f"📈 Predicted 7-day return for {ticker}: {predicted_pct}%")

    except Exception as e:
        print(f"⚠️ Error processing {ticker}: {e}")
