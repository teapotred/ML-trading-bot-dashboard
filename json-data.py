from alpaca.trading.client import TradingClient
import requests
import json

ALPACA_API_KEY = "PKMVVAY1RF8022U1ATQ3"
ALPACA_SECRET = 'r6GrduxcbuRrXUJlk3WPZJsQU08fAcxqemQYDg2I'
BASE_URL = 'https://paper-api.alpaca.markets/v2'

headers = {
    "APCA-API-KEY-ID": ALPACA_API_KEY,
    "APCA-API-SECRET-KEY": ALPACA_SECRET
}

TICKERS = ["LCID", "TSLA", "RIOT", "SOFI"]


for ticker in TICKERS:
    trading_client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET, paper=True)
    asset = trading_client.get_asset(ticker)
    print(f"Symbol: {ticker}, Tradable: {asset.tradable}, Status: {asset.status}, Class: {asset.asset_class}")
