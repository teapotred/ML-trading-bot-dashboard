from yahooquery import Ticker
import pandas as pd
import os

# === Configuration ===
tickers = ["BABA", "ROKU", "C", "F", "RIVN", "CHPT"]
start_date = "2010-01-01"
end_date = "2024-12-31"
output_folder = "ticker-yearly"

os.makedirs(output_folder, exist_ok=True)

for symbol in tickers:
    try:
        print(f"📥 Downloading {symbol}...")
        ticker = Ticker(symbol)
        hist = ticker.history(start=start_date, end=end_date)

        if hist.empty:
            print(f"⚠️ No data for {symbol}, skipping.")
            continue

        # Reset index to get 'date' column
        hist = hist.reset_index()
        hist = hist[hist['symbol'] == symbol]

        # Ensure required columns exist, fill if missing
        for col in ['dividends', 'splits']:
            if col not in hist.columns:
                hist[col] = 0.0  # or pd.NA if you want

        # Select and rename columns
        df = hist[['date', 'open', 'high', 'low', 'close', 'volume', 'dividends', 'splits']].copy()
        df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits']

        # Ensure timezone is kept in ISO format
        df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize('America/New_York', ambiguous='NaT', nonexistent='shift_forward')

        # Save CSV
        output_path = os.path.join(output_folder, f"{symbol}_yearly.csv")
        df.to_csv(output_path, index=False)
        print(f"✅ Saved: {output_path}")

    except Exception as e:
        print(f"❌ Error with {symbol}: {e}")

print("\n✅ All available tickers processed.")
