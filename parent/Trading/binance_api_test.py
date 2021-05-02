# -*- coding: utf-8 -*-
"""
Created on Sat Aug  8 14:47:21 2020

@author: vinmue
"""
from binance.websockets import BinanceSocketManager
from binance.client import Client
#from poloniex import Poloniex
from datetime import datetime
import requests
import pandas as pd
from collections import ChainMap
from datetime import datetime
from BotChecker import BotChecker

checker=BotChecker()
pairs=checker.pairs

api_key="ScXClxYXs8AnGpB7X8LeDducznZVHOylFcDh5ETsAt4BQh9FXG4daC04xOdRjJ5w"
api_secret="N8aLvhbx55aaEQbErj89qESFGdfcZGx1YkNeCbbSnV9KwgSUDxmhdtyqWKdJctzu"
client = Client(api_key, api_secret)
order = client.create_test_order(
    symbol='BNBBTC',
    side=Client.SIDE_BUY,
    type=Client.ORDER_TYPE_MARKET,
    quantity=100)
prices = client.get_all_tickers()
#conn=Poloniex('apikey','secret')



#print('order:')
#print(order)
#print('prices:')
#print(prices)

#request=requests.get("https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT")

request=requests.get("https://api.binance.com/api/v3/ticker/price")
ticker=request.json()
dic= {each.pop('symbol'): each for each in ticker}  
print(dic.keys())



start=datetime(2020,6,1)
end=datetime(2020,6,6)
start=datetime.timestamp(start)*1000
end=datetime.timestamp(end)*1000
print(start)
print(end)
market = 'BTCUSDT'
tick_interval = '1h'
problematic=[]

url = 'https://api.binance.com/api/v1/klines?symbol=XRPBTC&interval=5m&startTime={}&endTime={}'.format( int(start),int(end))
request=requests.get(url)
request_json=request.json()
print(request_json) 

    
'''
df=pd.DataFrame(ticker)
print(df)
ticker=df.to_dict()
print(ticker)
'''



'''
conn=Poloniex('apikey','secret')
print('prices:')
#print(conn.returnTicker())
print('chart:')
start=datetime(2020,7,10)
end=datetime(2020,8,10)
start=datetime.timestamp(start)
end=datetime.timestamp(end)
print(conn.returnChartData('USDT_BTC', 300,start,end))
'''



'''
from poloniex import Poloniex
from binance.websockets import BinanceSocketManager

client = Client(api_key, api_secret)

# get market depth
depth = client.get_order_book(symbol='BNBBTC')

# place a test market buy order, to place an actual order use the create_order function
order = client.create_test_order(
    symbol='BNBBTC',
    side=Client.SIDE_BUY,
    type=Client.ORDER_TYPE_MARKET,
    quantity=100)

# get all symbol prices
prices = client.get_all_tickers()

# withdraw 100 ETH
# check docs for assumptions around withdrawals
from binance.exceptions import BinanceAPIException, BinanceWithdrawException
try:
    result = client.withdraw(
        asset='ETH',
        address='<eth_address>',
        amount=100)
except BinanceAPIException as e:
    print(e)
except BinanceWithdrawException as e:
    print(e)
else:
    print("Success")

# fetch list of withdrawals
withdraws = client.get_withdraw_history()

# fetch list of ETH withdrawals
eth_withdraws = client.get_withdraw_history(asset='ETH')

# get a deposit address for BTC
address = client.get_deposit_address(asset='BTC')


bm = BinanceSocketManager(client)
bm.start_aggtrade_socket('BNBBTC', process_message)
bm.start()

# get historical kline data from any date range

# fetch 1 minute klines for the last day up until now
klines = client.get_historical_klines("BNBBTC", Client.KLINE_INTERVAL_1MINUTE, "1 day ago UTC")

# fetch 30 minute klines for the last month of 2017
klines = client.get_historical_klines("ETHBTC", Client.KLINE_INTERVAL_30MINUTE, "1 Dec, 2017", "1 Jan, 2018")

# fetch weekly klines since it listed
klines = client.get_historical_klines("NEOBTC", Client.KLINE_INTERVAL_1WEEK, "1 Jan, 2017")
'''