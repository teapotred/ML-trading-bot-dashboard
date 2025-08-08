from yahooquery import Ticker
import pandas as pd
import os
from datetime import datetime, timedelta

# === Configuration ===
tickers = ["MARA", "SOFI", "RIOT", "LCID", "RIVN", "CHPT"]
end_date = datetime.now()
start_date = end_date - timedelta(days=5479)  # approx 15 years
output_folder = "ticker-yearly"

# Format dates as strings
start_date_str = start_date.strftime('%Y-%m-%d')
end_date_str = end_date.strftime('%Y-%m-%d')

os.makedirs(output_folder, exist_ok=True)

for symbol in tickers:
    try:
        print(f"📥 Downloading {symbol} ({start_date_str} to {end_date_str})...")
        ticker = Ticker(symbol)
        hist = ticker.history(start=start_date_str, end=end_date_str)

        if hist.empty:
            print(f"⚠️ No data for {symbol}, skipping.")
            continue

        hist = hist.reset_index()
        hist = hist[hist['symbol'] == symbol]

        for col in ['dividends', 'splits']:
            if col not in hist.columns:
                hist[col] = 0.0

        df = hist[['date', 'open', 'high', 'low', 'close', 'volume', 'dividends', 'splits']].copy()
        df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits']

        # Standardize timezone to New York
        df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize('America/New_York')

        # Save file
        output_path = os.path.join(output_folder, f"{symbol}_yearly.csv")
        df.to_csv(output_path, index=False)
        print(f"✅ Saved: {output_path}")

    except Exception as e:
        print(f"❌ Error with {symbol}: {e}")

print("\n✅ All tickers processed.")
