import pandas as pd
from alpaca.trading.client import TradingClient

# CONFIG
API_KEY = "PK6EBSHGW3JJTUUH9898"
API_SECRET = "WAWaJVPMCJ7Yx7dFeGpQUzR5UIvxhY8oYik6xsih"
USE_PAPER = True
LOG_FILE = "important-files/trade_log.csv"

# Alpaca client
client = TradingClient(API_KEY, API_SECRET, paper=USE_PAPER)

# Read log
df = pd.read_csv(LOG_FILE)

# Analyze each trade
results = []
for _, row in df.iterrows():
    try:
        order = client.get_order_by_id(row['OrderID'])
        entry_price = float(row['Price'])
        qty = float(row['Qty'])
        filled_price = float(order.filled_avg_price) if order.filled_avg_price else None
        status = order.status

        profit = round((filled_price - entry_price) * qty, 2) if filled_price and status == "filled" else None

        results.append({
            "Date": row['DateTime'],
            "Symbol": row['Symbol'],
            "Qty": qty,
            "Entry Price": entry_price,
            "Filled Price": filled_price,
            "Status": status,
            "Profit/Loss": profit
        })
    except Exception as e:
        results.append({
            "Date": row['DateTime'],
            "Symbol": row['Symbol'],
            "Qty": row['Qty'],
            "Entry Price": row['Price'],
            "Filled Price": None,
            "Status": f"Error: {e}",
            "Profit/Loss": None
        })

# Save or print
results_df = pd.DataFrame(results)
results_df.to_csv("important-files/trade_results_summary.csv", index=False)
print(results_df)
