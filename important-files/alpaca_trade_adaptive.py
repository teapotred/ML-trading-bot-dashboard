import os
import csv
from datetime import datetime
import numpy as np

import pandas as pd
import xgboost as xgb
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import (
    LimitOrderRequest, TakeProfitRequest, StopLossRequest
)

# ========= CONFIG =========
API_KEY    = "PK0WNH4PZXTQ2KHRN089"
API_SECRET = "mziJNe7Rgb9qKWVlUgTQI9yv5CAXniCqlrlZbqvg"
USE_PAPER  = True

DATA_DIR   = "ticker-yearly"
MODEL_DIR  = "xgb_models"
TICKERS    = ['SOFI','RIOT','CHPT','LCID','RIVN','MARA']

ORDER_QTY        = 3
STOP_LOSS_PCT    = 0.02     # 2%
TAKE_PROFIT_PCT  = 0.04     # 4%

# Prediction thresholds
BUY_THRESH   = 0.55
SHORT_THRESH = 0.45

# Entry style: "limit_buffer" (recommended), "limit_strict"
ENTRY_MODE = "limit_buffer"

# Buffer for marketable limit entries (auto-adjusted by price)
# Under $5 -> 0.50%; $5–$20 -> 0.25%; $20+ -> 0.10%
def entry_buffer_pct(price):
    if price < 5:
        return 0.005
    elif price < 20:
        return 0.0025
    else:
        return 0.001

# Logging
PREDICTION_LOG = "important-files/predictions_log.csv"   # per-run prediction probs
TRADE_LOG      = "important-files/trade_log.csv"         # orders placed

os.makedirs("important-files", exist_ok=True)

# ========= UTILITIES =========
def round_cents(x: float) -> float:
    # Always 2 decimals, avoid sub-penny rejections
    return float(f"{x:.2f}")

def ensure_logs_exist():
    if not os.path.isfile(PREDICTION_LOG):
        with open(PREDICTION_LOG, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["DateTime","Symbol","ProbUp","Close","Mode"])
    if not os.path.isfile(TRADE_LOG):
        with open(TRADE_LOG, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["DateTime","Symbol","Prediction","Qty","Price","StopLoss","TakeProfit","OrderID"])

def log_prediction(symbol, prob_up, close, mode):
    with open(PREDICTION_LOG, "a", newline="") as f:
        w = csv.writer(f)
        w.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), symbol, f"{prob_up:.6f}", close, mode])

def log_trade(symbol, prediction, qty, current_price, stop_loss, take_profit, order_id):
    with open(TRADE_LOG, "a", newline="") as f:
        w = csv.writer(f)
        w.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    symbol, prediction, qty, current_price, stop_loss, take_profit, order_id])

# ========= FEATURE ENGINEERING (match your training) =========
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df[['Open','High','Low','Close','Volume']].copy()
    # --- original features ---
    df['SMA20'] = df['Close'].rolling(window=20).mean()
    df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['RSI']   = df['Close'].pct_change().rolling(14).apply(
        lambda x: (x[x>0].sum()/abs(x).sum())*100 if abs(x).sum()!=0 else 0
    )
    df['MACD']  = df['Close'].ewm(span=12).mean() - df['Close'].ewm(span=26).mean()
    df['ATR']   = (df['High'] - df['Low']).rolling(14).mean()
    df['OBV']   = (df['Volume'] * (df['Close'].diff() > 0).map({True:1, False:-1})).cumsum()

    for i in range(1,6):
        df[f'return_lag_{i}'] = df['Close'].pct_change(i).shift(1)

    df['rolling_std']   = df['Close'].rolling(window=5).std()
    df['rolling_mean']  = df['Close'].rolling(window=5).mean()
    df['Close/Open']    = df['Close']/df['Open']
    df['High/Low']      = df['High']/df['Low']
    df['Momentum']      = df['Close'] - df['Close'].shift(3)
    df['Volatility']    = df['Close'].rolling(window=5).std()
    df['VolumeChange']  = df['Volume'].pct_change()

    rolling_mean = df['Close'].rolling(window=20).mean()
    rolling_std  = df['Close'].rolling(window=20).std()
    df['BB_Width'] = (rolling_mean + 2*rolling_std) - (rolling_mean - 2*rolling_std)

    # --- NEW features (to match training) ---
    df['ExpVolatility'] = df['Close'].ewm(span=10, adjust=False).std()
    df['ROC']           = df['Close'].pct_change(periods=5)

    low_min  = df['Low'].rolling(window=14).min()
    high_max = df['High'].rolling(window=14).max()
    denom = (high_max - low_min).replace(0, np.nan)  # avoid div/0
    df['%K'] = 100 * ((df['Close'] - low_min) / denom)
    df['%D'] = df['%K'].rolling(window=3).mean()

    df['Body']      = (df['Close'] - df['Open']).abs()
    df['UpperWick'] = df['High'] - df[['Close','Open']].max(axis=1)
    df['LowerWick'] = df[['Close','Open']].min(axis=1) - df['Low']
    df['RVOL']      = df['Volume'] / df['Volume'].rolling(window=20).mean()
    df['Trend']     = df['Close'].diff().rolling(window=5).sum()

    # final clean
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    return df

FEATURES = [
    'SMA20','EMA20','RSI','MACD','ATR','OBV',
    'rolling_std','rolling_mean','Close/Open','High/Low',
    'Momentum','Volatility','VolumeChange','BB_Width',
    'ExpVolatility','ROC','%K','%D','Body','UpperWick','LowerWick','RVOL','Trend'
] + [f'return_lag_{i}' for i in range(1,6)]

# ========= MAIN =========
def main():
    ensure_logs_exist()
    client = TradingClient(API_KEY, API_SECRET, paper=USE_PAPER)

    for ticker in TICKERS:
        print(f"\n📊 Processing {ticker}...")
        file_path  = os.path.join(DATA_DIR, f"{ticker}_yearly.csv")
        model_path = os.path.join(MODEL_DIR, f"{ticker}_model_v1.json")

        if not os.path.exists(file_path):
            print(f"❌ File not found for {ticker}. Skipping.")
            continue
        if not os.path.exists(model_path):
            print(f"❌ Model not found for {ticker}. Skipping.")
            continue

        try:
            raw = pd.read_csv(file_path)
            df  = build_features(raw)
            X   = df[FEATURES]
            x_now = X.iloc[-1:].copy()

            model = xgb.XGBClassifier()
            model.load_model(model_path)
            prob_up = float(model.predict_proba(x_now)[0][1])
            last_close = float(df['Close'].iloc[-1])

            # Log the prediction either way
            log_prediction(ticker, prob_up, round_cents(last_close), ENTRY_MODE)

            # Decide direction
            side = None
            if prob_up >= BUY_THRESH:
                side = OrderSide.BUY
            elif prob_up <= SHORT_THRESH:
                side = OrderSide.SELL

            if side is None:
                print(f"⏸️ {ticker} neutral ({prob_up:.3f}). No trade.")
                continue

            # Compute entry price (marketable limit with buffer)
            buf = entry_buffer_pct(last_close)
            if ENTRY_MODE == "limit_buffer":
                if side == OrderSide.BUY:
                    entry_price = last_close * (1 + buf)     # pay up a hair to get filled
                else:  # short
                    entry_price = last_close * (1 - buf)     # sell slightly below to get filled
            else:
                entry_price = last_close

            entry_price = round_cents(entry_price)

            # Bracket TP / SL based on the entry price we’re submitting
            if side == OrderSide.BUY:
                tp = round_cents(entry_price * (1 + TAKE_PROFIT_PCT))
                sl = round_cents(entry_price * (1 - STOP_LOSS_PCT))
                order_data = LimitOrderRequest(
                    symbol=ticker,
                    qty=ORDER_QTY,
                    side=OrderSide.BUY,
                    time_in_force=TimeInForce.GTC,
                    order_class='bracket',
                    limit_price=entry_price,
                    take_profit=TakeProfitRequest(limit_price=tp),
                    stop_loss=StopLossRequest(stop_price=sl),
                )
            else:
                # SHORT: TP below, SL above; SL must be >= base + 0.01
                tp = round_cents(entry_price * (1 - TAKE_PROFIT_PCT))
                theoretical_sl = entry_price * (1 + STOP_LOSS_PCT)
                min_allowed    = entry_price + 0.01
                sl = round_cents(max(theoretical_sl, min_allowed))

                order_data = LimitOrderRequest(
                    symbol=ticker,
                    qty=ORDER_QTY,
                    side=OrderSide.SELL,
                    time_in_force=TimeInForce.GTC,
                    order_class='bracket',
                    limit_price=entry_price,
                    take_profit=TakeProfitRequest(limit_price=tp),
                    stop_loss=StopLossRequest(stop_price=sl),
                )

            order = client.submit_order(order_data)
            print(f"📥 {ticker} {('BUY' if side==OrderSide.BUY else 'SHORT')} placed @ {entry_price} | id={order.id}")

            # Log order for P/L script
            log_trade(
                symbol=ticker,
                prediction=prob_up,
                qty=ORDER_QTY,
                current_price=entry_price,
                stop_loss=sl,
                take_profit=tp,
                order_id=order.id
            )

        except Exception as e:
            print(f"⚠️ Error processing {ticker}: {e}")

    print("\n🧾 Predictions saved to", PREDICTION_LOG)
    print("🧾 Orders (if any) saved to", TRADE_LOG)


if __name__ == "__main__":
    main()
