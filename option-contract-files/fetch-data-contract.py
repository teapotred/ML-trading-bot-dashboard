import requests
import pandas as pd
from datetime import datetime
import os

# Config
API_KEY = "PK6EBSHGW3JJTUUH9898"
API_SECRET = "WAWaJVPMCJ7Yx7dFeGpQUzR5UIvxhY8oYik6xsih"
BASE_URL = "https://paper-api.alpaca.markets/v2/options/contracts"
SAVE_FOLDER = "option-contract-files"
os.makedirs(SAVE_FOLDER, exist_ok=True)

symbols = ['AAPL', 'SPY', 'RIOT', 'RIVN', 'LCID', 'TSLA']

headers = {
    "accept": "application/json",
    "APCA-API-KEY-ID": API_KEY,
    "APCA-API-SECRET-KEY": API_SECRET
}

for symbol in symbols:
    print(f"🔍 Fetching contracts for {symbol}")
    all_contracts = []
    next_token = None

    while True:
        params = {
            "underlying_symbols": symbol,
        }
        if next_token:
            params["page_token"] = next_token

        try:
            response = requests.get(BASE_URL, headers=headers, params=params)
            response.raise_for_status()
            data = response.json()

            contracts = data.get("option_contracts", [])
            all_contracts.extend(contracts)

            next_token = data.get("next_page_token")
            if not next_token:
                break  # No more pages

        except Exception as e:
            print(f"❌ Error for {symbol}: {e}")
            break

    if all_contracts:
        df = pd.DataFrame(all_contracts)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"{SAVE_FOLDER}/{symbol}_options_{timestamp}.csv"
        df.to_csv(filename, index=False)
        print(f"✅ Saved {len(df)} contracts for {symbol} to {filename}")
    else:
        print(f"⚠️ No contracts found for {symbol}")
