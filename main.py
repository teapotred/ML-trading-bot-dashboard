import pandas as pd
import yfinance as yf
from ta.momentum import RSIIndicator
from alpaca_trade_api.rest import REST, TimeFrame
import time

# === SETUP ===
API_KEY = 'https://paper-api.alpaca.markets/v2'
API_SECRET = 'auoASnGtMeLF7JZ9z0117QnI3byNgfCpwxEXohnG'
BASE_URL = 'https://paper-api.alpaca.markets'  # Use paper trading

api = REST(API_KEY, API_SECRET, BASE_URL)

symbol = 'NVDA'
qty = 1

# === FETCH DATA ===
df = yf.download(symbol, period='30d', interval='1d')
df['RSI'] = RSIIndicator(df['Close'], window=14).rsi()

latest_rsi = df['RSI'].iloc[-1]
latest_price = df['Close'].iloc[-1]

print(f"{symbol} RSI: {latest_rsi:.2f} | Price: ${latest_price:.2f}")

# === DECISION LOGIC ===
position = None
try:
    position = api.get_position(symbol)
    has_position = True
except:
    has_position = False

if latest_rsi < 30 and not has_position:
    print("RSI < 30 → Buying")
    api.submit_order(symbol=symbol, qty=qty, side='buy', type='market', time_in_force='gtc')

elif latest_rsi > 70 and has_position:
    print("RSI > 70 → Selling")
    api.submit_order(symbol=symbol, qty=qty, side='sell', type='market', time_in_force='gtc')
else:
    print("No action today")
