from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
from alpaca.trading.requests import OrderSide, TimeInForce


API_KEY= "PK0WNH4PZXTQ2KHRN089"
API_SECRET ="mziJNe7Rgb9qKWVlUgTQI9yv5CAXniCqlrlZbqvg"

tradingClient = TradingClient(API_KEY, API_SECRET, paper=True)

positions = tradingClient.get_open_position('SPY')
print(positions)


TICKERS = ['MARA']
#assets = tradingClient.get_all_assets()

#for asset in assets:
#    if asset.symbol == 'SPY':
#        print(asset.tradable)

#simple order req, gets stock at whatever price it is at
#btw it does work lol
# market_order_data = MarketOrderRequest(
#     symbol="SPY",
#     qty=1,
#     side=OrderSide.BUY,
#     time_in_force= TimeInForce.DAY
#     )
# 
# market_order = tradingClient.submit_order(market_order_data)
# print(market_order)

#limit orders
