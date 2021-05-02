import time

# for live trading, currently out of use


class BotCandlestick(object):
    def __init__(self, period, log, checker, date=None):
        self.date = date
        self.startTime = time.time()
        self.period = period
        self.output = log
        self.candlestick = {}
        self.open = {}
        self.close = {}
        self.high = {}
        self.low = {}
        self.volume = {}
        self.current = {}
        self.counter = {}
        self.pairs = checker.pairs
        for pair in self.pairs:
            self.candlestick[pair] = {}
            self.open[pair] = None
            self.close[pair] = None
            self.high[pair] = None
            self.low[pair] = None
            self.volume[pair] = None
            self.current[pair] = None
            self.counter[pair] = 0

    def tick(self, pair, prices):
        self.counter[pair] += 1
        self.current[pair] = float(prices[pair]['price'])
        if self.open[pair] is None:
            self.open[pair] = self.current[pair]
        if (self.high[pair] is None) or (self.current[pair] > self.high[pair]):
            self.high[pair] = self.current[pair]
        if (self.low[pair] is None) or (self.current[pair] < self.low[pair]):
            self.low[pair] = self.current[pair]
        if time.time() >= (self.startTime + self.period):
            self.close[pair] = self.current[pair]

        #self.output.log("Pair: " + pair + " Open: " + str(self.open[pair]) + " Close: " + str(self.close[pair]) + " High: " + str(self.high[pair]) +
                        #" Low: " + str(self.low[pair]) + " Current: " + str(self.current[pair]))
        self.candlestick[pair]['open'] = self.open[pair]
        self.candlestick[pair]['close'] = self.close[pair]
        self.candlestick[pair]['low'] = self.low[pair]
        self.candlestick[pair]['high'] = self.high[pair]

    def isClosed(self, pair):
        if self.close[pair] is not None:
            return True
        else:
            return False
