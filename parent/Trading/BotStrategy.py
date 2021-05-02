import sys
from BotRLTester import BotRLTester
from BotAPI import BotAPI
from BotTrade import BotTrade
from BotChecker import BotChecker
from BotChart import BotChart
#from RL import DeepQLearning

class BotStrategy(object):
    instance = None
    @staticmethod
    def get_instance(*args, **kwargs):
        if BotStrategy.instance is None:
            BotStrategy.instance = BotStrategy(*args, **kwargs)
            return BotStrategy.instance
        else:
            return BotStrategy.instance
    def __init__(self, period,trading, startBalance=0.01):
        if not BotStrategy.instance is None:
            raise RuntimeError("multiple instances must not be created of this class")
        checker = BotChecker.get_instance()
        self.pairs = checker.pairs
        self.coins = checker.coins

        self.api = BotAPI.get_instance()
        self.chart = BotChart.get_instance()
        self.trade = BotTrade(period=period,startBalance=startBalance)

        self.trading=trading
        print("Real deals are made: "+ "yes" if trading else "no")
        self.oldValue = 0
        self.ovrValue = 0
        self.period = period

        self.botRLTester = BotRLTester()
        self.AI_trained = True
        BotChecker.instance = self


    def evaluatePosition(self, pos):
        buyOptions = []
        sellOptions = []
        print(self.chart.history.to_numpy().shape)
        if self.trading:
            self.realDeal(buyOptions,sellOptions, pos)
        else:
            self.sim(buyOptions,sellOptions, pos)
        self.trade.giveInfo()



    def usePPO(self, pos):
        if not self.AI_trained:
            self.botRLTester.train()
            self.AI_trained = True
        buyOptions = []
        sellOptions = []
        print(self.chart.history.to_numpy().shape)
        if self.trading:
            self.realDeal(buyOptions, sellOptions, pos)
        else:
            self.sim(buyOptions, sellOptions, pos)
        self.trade.giveInfo()

    def sim(self, sellOptions:list,buyOptions:list, pos):
        self.trade.simSell(sellOptions, pos)
        self.trade.simBuy(buyOptions, pos)
        self.trade.simBalance(pos)
        return  self.trade.oldValue, self.trade.ovrValue
    def realDeal(self,sellOptions:list, buyOptions:list):
        self.trade.Sell(sellOptions)
        self.trade.Buy(buyOptions)
        self.trade.balance()
        return self.trade.oldValue, self.trade.ovrValue