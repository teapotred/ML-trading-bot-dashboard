# trade_logger.py
import csv
from datetime import datetime
import os

LOG_FILE = "trade_log.csv"

def log_trade(symbol, prediction, qty, current_price, stop_loss, take_profit, order_id):
    file_exists = os.path.isfile(LOG_FILE)
    with open(LOG_FILE, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["DateTime", "Symbol", "Prediction", "Qty", "Price", "StopLoss", "TakeProfit", "OrderID"])
        writer.writerow([
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            symbol, prediction, qty, current_price, stop_loss, take_profit, order_id
        ])
