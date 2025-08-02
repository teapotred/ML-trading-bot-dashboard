import os
import pandas as pd
import xgboost as xgb
from datetime import datetime
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (
    LimitOrderRequest, TakeProfitRequest, StopLossRequest
)
from alpaca.trading.enums import OrderSide, TimeInForce
from trade_logger import log_trade

# === CONFIG ===
API_KEY = "PK6EBSHGW3JJTUUH9898"
API_SECRET = "WAWaJVPMCJ7Yx7dFeGpQUzR5UIvxhY8oYik6xsih"
USE_PAPER = True
DATA_DIR = "ticker-yearly"
MODEL_DIR = "xgb_models"
TICKERS = ['LCID', 'RIOT', 'SOFI', 'SPY', 'TSLA', "RIVN", "CHPT"]
STOP_LOSS_PCT = 0.02
TAKE_PROFIT_PCT = 0.04
ORDER_QTY = 1
PREDICTION_LOG = f"logs/trade_log.csv"

# === INIT ===
trading_client = TradingClient(API_KEY, API_SECRET, paper=USE_PAPER)
os.makedirs("logs", exist_ok=True)
prediction_records = []

# === LOOP THROUGH TICKERS ===
for ticker in TICKERS:
    print(f"\n📊 Processing {ticker}...")
    file_path = os.path.join(DATA_DIR, f"{ticker}_yearly.csv")
    model_path = os.path.join(MODEL_DIR, f"{ticker}_model_v1.json")

    if not os.path.exists(file_path):
        print(f"❌ File not found for {ticker}. Skipping.")
        continue
    if not os.path.exists(model_path):
        print(f"❌ Model not found for {ticker}. Skipping.")
        continue

    try:
        df = pd.read_csv(file_path)
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()

        # === Feature Engineering ===
        df['SMA20'] = df['Close'].rolling(window=20).mean()
        df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
        df['RSI'] = df['Close'].pct_change().rolling(14).apply(
            lambda x: (x[x > 0].sum() / abs(x).sum()) * 100 if abs(x).sum() != 0 else 0
        )
        df['MACD'] = df['Close'].ewm(span=12).mean() - df['Close'].ewm(span=26).mean()
        df['ATR'] = (df['High'] - df['Low']).rolling(14).mean()
        df['OBV'] = (df['Volume'] * (df['Close'].diff() > 0).map({True: 1, False: -1})).cumsum()

        for i in range(1, 6):
            df[f'return_lag_{i}'] = df['Close'].pct_change(i).shift(1)

        df['rolling_std'] = df['Close'].rolling(window=5).std()
        df['rolling_mean'] = df['Close'].rolling(window=5).mean()
        df['Close/Open'] = df['Close'] / df['Open']
        df['High/Low'] = df['High'] / df['Low']
        df['Momentum'] = df['Close'] - df['Close'].shift(3)
        df['Volatility'] = df['Close'].rolling(window=5).std()
        df['VolumeChange'] = df['Volume'].pct_change()
        df['BB_Width'] = (
            df['Close'].rolling(window=20).mean() + 2 * df['Close'].rolling(20).std()
        ) - (
            df['Close'].rolling(window=20).mean() - 2 * df['Close'].rolling(20).std()
        )

        df.dropna(inplace=True)

        features = ['SMA20', 'EMA20', 'RSI', 'MACD', 'ATR', 'OBV',
                    'rolling_std', 'rolling_mean', 'Close/Open', 'High/Low',
                    'Momentum', 'Volatility', 'VolumeChange', 'BB_Width'] + \
                   [f'return_lag_{i}' for i in range(1, 6)]

        X = df[features]
        latest_features = X.iloc[-1:]

        model = xgb.XGBClassifier()
        model.load_model(model_path)
        prob = model.predict_proba(latest_features)[0][1]
        current_price = round(df['Close'].iloc[-1], 2)

        prediction_records.append({
            "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "ticker": ticker,
            "prob_up": round(prob, 4),
            "close": current_price
        })

        # === BUY ===
        if prob > 0.55:
            print(f"✅ {ticker} predicted to go UP. Placing buy order.")
            order_data = LimitOrderRequest(
                symbol=ticker,
                qty=ORDER_QTY,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.GTC,
                order_class='bracket',
                limit_price=current_price,
                take_profit=TakeProfitRequest(limit_price=round(current_price * (1 + TAKE_PROFIT_PCT), 2)),
                stop_loss=StopLossRequest(stop_price=round(current_price * (1 - STOP_LOSS_PCT), 2)),
            )
            order = trading_client.submit_order(order_data)
            print(f"📥 Order placed: {order.id}")
            log_trade(ticker, prob, ORDER_QTY, current_price,
                      round(current_price * (1 - STOP_LOSS_PCT), 2),
                      round(current_price * (1 + TAKE_PROFIT_PCT), 2),
                      order.id)

        # === SHORT ===
        elif prob < 0.45:
            print(f"🔻 {ticker} predicted to go DOWN. Placing short order.")
            stop_price = round(max(current_price + 0.01, current_price * (1 + STOP_LOSS_PCT)), 2)
            tp_price = round(current_price * (1 - TAKE_PROFIT_PCT), 2)

            order_data = LimitOrderRequest(
                symbol=ticker,
                qty=ORDER_QTY,
                side=OrderSide.SELL,
                time_in_force=TimeInForce.GTC,
                order_class='bracket',
                limit_price=current_price,
                take_profit=TakeProfitRequest(limit_price=tp_price),
                stop_loss=StopLossRequest(stop_price=stop_price),
            )
            order = trading_client.submit_order(order_data)
            print(f"📥 Short order placed: {order.id}")
            log_trade(ticker, prob, ORDER_QTY, current_price, stop_price, tp_price, order.id)

        else:
            print(f"⏸️ {ticker} has neutral prediction ({round(prob, 3)}). No trade.")

    except Exception as e:
        print(f"⚠️ Error processing {ticker}: {e}")

# === Save all prediction probabilities to CSV ===
pd.DataFrame(prediction_records).to_csv(PREDICTION_LOG, index=False)
print(f"\n🧾 Predictions saved to {PREDICTION_LOG}")
