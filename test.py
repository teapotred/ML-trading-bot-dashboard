from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
from alpaca.trading.requests import OrderSide, TimeInForce


api_key = "PK6EBSHGW3JJTUUH9898"
secret_key = "WAWaJVPMCJ7Yx7dFeGpQUzR5UIvxhY8oYik6xsih"
tradingClient = TradingClient(api_key, secret_key, paper=True)

positions = tradingClient.get_open_position('SPY')
print(positions)


#TICKERS = ["LCID", "TSLA", "RIOT", "SOFI", "AAPL", "GOOG", "TEMP", "XYZ",'LCID','TSLA','RIOT','INTC','SOFI','MARA','WBD','WMT','IBM','AMD','RBLX','NEE','META','AAPL','SPY','NVDA']
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
