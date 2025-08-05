import requests
import pandas as pd
from datetime import datetime

# CONFIG
API_KEY = "PK6EBSHGW3JJTUUH9898"
API_SECRET = "WAWaJVPMCJ7Yx7dFeGpQUzR5UIvxhY8oYik6xsih"
BASE_URL = "https://paper-api.alpaca.markets"
HEADERS = {
    "APCA-API-KEY-ID": API_KEY,
    "APCA-API-SECRET-KEY": API_SECRET
}

# Today's date
today = datetime.now().strftime("%Y-%m-%d")

# Call the account activities endpoint
url = f"{BASE_URL}/v2/account/activities/FILL?date={today}"
response = requests.get(url, headers=HEADERS)
response.raise_for_status()

data = response.json()

# Convert to DataFrame
df = pd.DataFrame(data)

if df.empty:
    print("📭 No fill activities today.")
else:
    df = df[["symbol", "qty", "side", "price", "Date"]]
    print("📈 Today's Filled Trades:")
    print(df)
    df.to_csv("important-files/todays_fills.csv", index=False)
