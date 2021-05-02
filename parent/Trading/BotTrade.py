from BotAPI import BotAPI

from BotChecker import BotChecker
from BotChart import BotChart
from colorit import color_front

class BotTrade(object):
    instance = None
    @staticmethod
    def get_instance( *args, **kwargs):
        if BotTrade.instance is None:
            BotTrade.instance = BotTrade(*args, **kwargs)
            return BotTrade.instance
        else:
            return BotTrade.instance
    def __init__(self,period,startBalance):
        if not BotTrade.instance is None:
            raise RuntimeError("multiple instances must not be created of this class")
        self.list_of_trades=[]
        self.notTraded=[]
        self.api=BotAPI.get_instance()
        self.counter=0
        self.chart=BotChart.get_instance()
        self.oldValue=0
        self.ovrValue=0
        checker = BotChecker.get_instance()
        self.pairs=checker.pairs
        self.coins=checker.coins
        self.period=period
        self.fee = {}
        self.fee['BTC'] = self.api.getFee('ETHBTC')
        self.fee['BNB'] = self.api.getFee('BNBBTC')
        self.fee['USDT'] = self.api.getFee('BTCUSDT')
        self.Yield=0
        self.balances = self.api.getBalance()
        self.simBalances={ coin: 0 for coin in self.coins}
        self.simBalances['BTC']=1/self.api.returnCurrentPrice("BTCUSDT")
        BotTrade.instance = self

    def Sell(self,definitiveSellOptions):

        self.counter=0
        for sell in definitiveSellOptions:
            if sell == "BTCUSDT":
                continue
            other = sell[:-3]  # other currency
            quantityBTC = self.balances[other]
            print(f"sell {sell}")
            quantity = self.api.filterSell(quantity=quantityBTC, pair=sell)
            if quantity:
                self.counter += 1

                status, sellPrice = self.api.Sell(pair=sell, quantity=quantity)
                if status == "FILLED":
                    self.list_of_trades.append(status)
                    print(f'sold {sell} at {sellPrice}')
                else:
                    self.notTraded.append(status, sell)
                    print(status, sell)

            else:
                print(f"Would have sold but no coins of this currency, {sell}")

    def Buy(self,definitiveBuyOptions):
        counter=0
        print(f"Buy options look like: {definitiveBuyOptions} ")
        for i, buy in enumerate(definitiveBuyOptions):
            number_of_buys = len(definitiveBuyOptions)
            fraction = 1 / number_of_buys
            if buy == "BTCUSDT":
                continue
            other = buy[:-3]  # other currency
            quantityBTC = self.balances['BTC'] * fraction

            if "BNB" not in ['BTC', other]:
                quantityBTC = quantityBTC * (1 - self.fee['BTC'])
            else:
                quantityBTC = quantityBTC * (1 - self.fee['BNB'])

            print(f"quantityTemp is: {quantityBTC}")
            quantity = self.api.filterBuy(quantityBTCstart=quantityBTC, pair=buy)

            if quantity:
                counter += 1

                status, buyPrice = self.api.Buy(pair=buy, quantity=quantity)
                if status == "FILLED":
                    self.list_of_trades.append(status)
                    print(f'bought {buy} at {buyPrice}')
                else:
                    self.notTraded.append(status, buy)
                    print(status, buy)

            else:
                print(f"Would have bought but no coins (BTC (or USDT)) to buy this currency, {buy}")

        # evaluate the portfolio value called overall
    def balance(self):
        self.oldValue = self.ovrValue
        self.ovrValue = 0
        self.balances = self.api.getBalance()

        for pair in self.pairs:
            if pair == 'BTCUSDT':
                other = 'USDT'
            else:
                other = pair[:-3]  # other currency
            self.ovrValue += self.balances[other] * self.chart.history[pair+'_close'].iloc[-1] * self.chart.history['BTCUSDT_close'].iloc[-1] #TODO make panda conform
        self.ovrValue += self.balances['USDT']
        self.ovrValue += self.balances['BTC'] * self.chart.history['BTCUSDT_close'].iloc[-1]
        #print('USDT:', self.balances['USDT'], 'BTC:', self.balances['BTC'], 'overall value:', self.ovrValue)
        self.counter += self.period
        self.Yield = self.ovrValue - self.oldValue

    def simBuy(self,definitiveBuyOptions,pos):
        print(f"Buy options look like: {definitiveBuyOptions} ")
        for i, buy in enumerate(definitiveBuyOptions):
            number_of_buys = len(definitiveBuyOptions)
            fraction = 1 / number_of_buys
            if buy == "BTCUSDT":
                continue
            other = buy[:-3]  # other currency
            quantityBTC = self.simBalances['BTC'] * fraction
            price=self.chart.history[buy+'_close'].iloc[pos]
            quantity=quantityBTC/price
            ###calculate new balances
            self.simBalances['BTC']-=quantityBTC
            self.simBalances[other]+=quantity

        # evaluate the portfolio value called overall
    def simSell(self,definitiveSellOptions,pos):
        print(f"Sell options look like: {definitiveSellOptions} ")
        for i, sell in enumerate(definitiveSellOptions):
            if sell== "BTCUSDT":
                continue
            other = sell[:-3]
            quantity=self.simBalances[other]
            price=self.chart.history[sell+'_close'].iloc[pos]
            quantityBTC=quantity*price
            self.simBalances['BTC']+=quantityBTC
            self.simBalances[other]=0
    def simBalance(self,pos):
        self.oldValue = self.ovrValue
        self.ovrValue = 0
        for pair in self.pairs:
            if pair == 'BTCUSDT':
                continue
            else:
                other = pair[:-3]  # other currency
            self.ovrValue += self.simBalances[other] * self.chart.history[pair + '_close'].iloc[pos] * \
                             self.chart.history['BTCUSDT_close'].iloc[pos]  # TODO make panda conform
        self.ovrValue += self.simBalances['BTC'] * self.chart.history['BTCUSDT_close'].iloc[pos]
        print("self.ovrValue: ", self.ovrValue)
        #print('BTC:', self.simBalances['BTC'], 'overall value:', self.ovrValue)
        self.counter += self.period
        self.Yield = self.ovrValue - self.oldValue
        return self.ovrValue

    def giveInfo(self):
        if self.Yield > 1:
            print(
                f"The {color_front('overall',55,114,253)} Value is: {color_front(str(self.ovrValue),55,114,253)}."
                  f" the {color_front('Yield',0,153,76)} is: {color_front(str(self.Yield), 0,153,76)}")
        elif self.Yield > 0:
            print(
                f"The {color_front('overall', 55, 114, 253)} Value is: {color_front(str(self.ovrValue), 55, 114, 253)}."
                f" the {color_front('Yield', 0,255,0)} is: {color_front(str(self.Yield), 0,255,0)}")
        elif self.Yield < -1:
            print(
                f"The {color_front('overall', 55, 114, 253)} Value is: {color_front(str(self.ovrValue), 55, 114, 253)}."
                f" the {color_front('Yield', 255,0,0)} is: {color_front(str(self.Yield), 255,0,0)}")
        else:
            print(
                f"The {color_front('overall', 55, 114, 253)} Value is: {color_front(str(self.ovrValue), 55, 114, 253)}."
                f" the {color_front('Yield', 255,128,0)} is: {color_front(str(self.Yield), 255,128,0)}")
        print("{} days passed".format(self.counter / 86400))

