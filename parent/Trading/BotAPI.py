# -*- coding: utf-8 -*-
"""
Created on Sat Aug  8 15:07:51 2020

@author: vinmue
"""
import requests
from BotChecker import BotChecker
import math
import time
from binance.client import Client
from decimal import Decimal
from decimal import getcontext
from decimal import ROUND_DOWN
import numpy as np


class BotAPI(object):
    instance = None
    @classmethod
    def get_instance(cls, *args, **kwargs):
        if not cls.instance:
            print("cls.instance:",cls.instance)
            cls.instance = BotAPI(*args, **kwargs)
            return BotAPI.instance
        else:
            return BotAPI.instance
    def __init__(self, period):
        if not BotAPI.instance is None:
            raise RuntimeError("multiple instances must not be created of this class")
        # from account yvieboy
        self.period = period

        self.api_key = "XXXXXXXXXXXXXXXXXX"
        self.api_secret = "XXXXXXXXXXXXXXXXX"

        self.client = Client(api_key=self.api_key, api_secret=self.api_secret)
        # for changing periods in seconds binance accepted strings
        self.period_dict = {60: "1m", 180: "3m", 300: "5m", 900: "15m", 1800: "30m", 3600: "1h", \
                            7200: "2h", 14400: "4h", 21600: "6h", 28800: "8h", 43200: "12h", 86400: "1d", \
                            259200: "3d", 1209600: "1w"}
        self.klineColumns = np.array([1, 2, 3, 4, 5])
        checker = BotChecker.get_instance()
        self.pairs = checker.pairs
        BotAPI.instance = self
    def getFee(self, pair):
        fee = self.client.get_trade_fee(symbol=pair)["tradeFee"][0]["taker"]
        return fee

    def returnTicker(self, period):
        chartLis = []
        columns = []
        for pair in self.pairs:
            period_str = self.period_dict[period]
            subStart = (time.time() - period) * 1000
            subEnd = (time.time()) * 1000
            url = f'https://api.binance.com/api/v3/klines?symbol={pair}&interval={period_str}&startTime={int(subStart)}&endTime={int(subEnd)}&limit=1000'
            for k in range(10):
                try:
                    # request = self.client.get_historical_klines(pair, period, int(subStart),
                    # int(subEnd), 1000)
                    request = requests.get(url, timeout=1)
                    request.raise_for_status()
                    break
                except Exception as Error:
                    print(Error)
                    print('request failed')
                    time.sleep(1)
            req_json = request.json()
            #print(req_json)
            # sometimes req_json is an empty list since the request hasn't worked out --> 8 additional seconds were sufficient
            chartLis += req_json[0][1:6]
        index = np.array(int(req_json[0][0] / 1000))
        return [chartLis],[index]

    def returnCurrentPrice(self, pair):
        # gives dictionary with keys symbol (pair) and price
        ###url="https://api.binance.com/api/v3/ticker/price?symbol={}".format(pair)
        for k in range(10):
            try:
                request = self.client.get_symbol_ticker(symbol=pair)
                ###request = requests.get(url, timeout=1)
                ###request.raise_for_status()
                break
            except:
                print('request failed')
                time.sleep(1)
        # request = requests.get("https://api.binance.com/api/v3/ticker/price?symbol={}".format(pair))
        currentPrice = request['price']
        return float(currentPrice)

    def returnChartData(self, pair, start, end, period):
        chartLis = []
        period_str = self.period_dict[period]
        # the pairs are written in poloniex convention with quote_base and therefore have to be reversed
        # binance works with timestamps in miliseconds so our timestamps have to be converted
        start = 1000 * start
        end = 1000 * end
        # split the request in chunks that have in maximum 1000 datapoints
        numParts = math.ceil((end - start) / (period * 1e6))
        print('pair:', pair, 'numparts:', numParts)
        for i in range(numParts):
            subStart = start + i * (end - start) / numParts
            subEnd = start + (i + 1) * (end - start) / numParts
            print('start:', start, 'end:', end, 'subStart:', subStart, 'subEnd:', subEnd)
            url = f'https://api.binance.com/api/v3/klines?symbol={pair}&interval={period_str}&startTime={int(subStart)}&endTime={int(subEnd)}&limit=1000'
            for k in range(10):
                try:
                    request = requests.get(url, timeout=1)
                    request.raise_for_status()
                    break
                except Exception as Error:
                    print(Error)
                    print('request failed')
                    time.sleep(1)
            chartLis.extend( request.json())
        # chart_lis is a list of lists, highest in hierarchy are the timestamps
        # chart becomes a list of dictionaries, a dictionary for each timestamp
        chartArray = np.array(chartLis,dtype=np.float64)
        print(chartArray.shape)
        '''
        try:
            result = np.array(list(map(int, chartArray[:, 0])))
        except:
            print(pair, chartArray)
        index = result / 1000
        '''
        data = chartArray[:, 1:6]
        index = (chartArray[:,0]/1000).astype(int)
        data=data.tolist()
        '''
        'date','open','high','low','close','volume','closeTime','quoteAssetVolume','numberOfTrades','takerBuyBaseAssetVolume','takerBuyAssetVolume'
        'date' = 0
        'open' = 1
        'high' = 2
        'low' = 3
        'close' = 4
        'volume' = 5
        'closeTime' = 6
        'quoteAssetVolume' =7
        'numberOfTrades' =8
        'takerBuyBaseAssetVolume' = 9
        'takerBuyQuoteAssetVolume' = 10
        '''
        return data,index

    def Buy(self, pair, quantity):

        print("buy: ", quantity, pair)
        for k in range(5):
            try:
                request = self.client.create_order(symbol=pair, side='buy', type='MARKET', quantity=quantity,
                                                   recvWindow=5000, newOrderRespType='FULL')
                break
            except Exception as error:
                print(f'request failed, {error}')
                if k >= 5:
                    return "CANCELED", 0
                time.sleep(1)
        print(f"the buy request looks like: {request}")
        status = request['status']
        buyPrice = float(request['price'])
        return status, buyPrice

    def Sell(self, pair, quantity):
        for k in range(5):
            try:
                request = self.client.create_order(symbol=pair, side='sell', type='MARKET', quantity=quantity,
                                                   recvWindow=5000, newOrderRespType='FULL')
                break
            except:
                print('request failed')
                time.sleep(1)

        print(f"the sell request looks like: {request}")
        status = request['status']
        sellPrice = float(request['price'])
        return status, sellPrice

    def getBalance(self):
        for k in range(10):
            try:
                request = self.client.get_account(recvWindow=5000)
                break
            except:
                print('request failed')
                time.sleep(1)
        # tempBalances is a list of dictionaries
        tempBalances = request['balances']
        balances = {}
        for dictionary in tempBalances:
            coin = dictionary['asset']
            balance = float(dictionary['free'])
            balances[coin] = balance
        return balances


    def filterBuy(self, pair, quantityBTCstart):
        # minNotional and priceFilter can also requested by get_symbol_info
        for i in range(10):
            try:
                info = self.client.get_symbol_info(symbol=pair)
                break
            except:
                time.sleep(1)
        minNotional = float(info['filters'][3]['minNotional'])

        for i in range(10):
            try:
                price = float(self.client.get_symbol_ticker(symbol=pair)["price"])
                break
            except:
                time.sleep(1)

        quantity = quantityBTCstart / price
        lotSize = info['filters'][2]
        minQty = float(lotSize['minQty'])
        stepSize = Decimal(lotSize['stepSize'])

        #getcontext().prec = 100
        getcontext().rounding = ROUND_DOWN
        quantity = stepSize * Decimal(math.floor(Decimal(quantity) / stepSize))
        quantityBTC = quantity * Decimal(price)
        print(f"quantity: {quantity} vs minQty: {minQty}")
        print(f"quantityBtc: {quantityBTC} vs minNotional: {minNotional}")
        if quantityBTC < minNotional or quantity < minQty:
            return False
        # quantity = Decimal(int(quantity * 1000000)) / 1000000
        return quantity

    def filterSell(self, pair, quantity):
        for i in range(10):
            try:
                info = self.client.get_symbol_info(symbol=pair)
                break
            except:
                time.sleep(1)
        lotSize = info['filters'][2]
        minQty = float(lotSize['minQty'])
        stepSize = Decimal(lotSize['stepSize'])
        getcontext().prec = 100
        getcontext().rounding = ROUND_DOWN
        quantity = stepSize * Decimal(math.floor(Decimal(quantity) / stepSize))
        minNotional = float(info['filters'][3]['minNotional'])

        for i in range(10):
            try:
                price = float(self.client.get_symbol_ticker(symbol=pair)["price"])
                break
            except:
                time.sleep(1)

        quantityBTC = quantity * Decimal(price)
        print(f"quantity: {quantity} vs minQty: {minQty}")
        print(f"quantityBtc: {quantityBTC} vs minNotional: {minNotional}")
        if quantity < minQty or quantityBTC < minNotional:
            return False
        print("filter: ", quantity, pair)
        return quantity
