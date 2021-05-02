import pandas as pd
import numpy as np
import time
from BotAPI import BotAPI
from BotChecker import BotChecker




class BotChart(object):
    instance = None
    @staticmethod
    def get_instance( *args, **kwargs):
        if BotChart.instance is None:
            BotChart.instance = BotChart(*args, **kwargs)
            return BotChart.instance
        else:
            return BotChart.instance
    def __init__(self, period):
        if not BotChart.instance is None:
            raise RuntimeError("multiple instances must not be created of this class")
        self.checker = BotChecker.get_instance()
        self.pairs = self.checker.pairs
        self.period = period
        self.api = BotAPI.get_instance()
        self.history = None
        self.counter = 0
        BotChart.instance = self


    def maxConsecutivesNaNs(self,df):
        self.counter = 0
        dfNaNs = df.isna()
        dfNaNs = dfNaNs.apply(self.countConsecutives, axis = 1)
        return dfNaNs.to_numpy().max()

    def countConsecutives(self, x):
        self.counter+=1
        self.counter*=x
        return self.counter

    def maxRatioNaNs(self, df):
        dfNaNs = df.isna()
        ratios = dfNaNs.sum(axis = 1)/len(dfNaNs.index)
        maxRatio = ratios.max()
        return maxRatio

    def appendData(self,data,index):

        oldIndex = self.history.index.copy(deep=True)
        index = pd.Int64Index(index)

        updatedIndex = oldIndex.append(index)
        updatedData=self.history.values.tolist()
        updatedData.extend(data)

        #########limit Dataframe growth########
        #updatedData = updatedData[-10000:] ###4500 is sufficient for 2 weeks of data in period 300
        #updatedIndex = updatedIndex[-10000:]
        ##################################
        updatedData = np.array(updatedData,dtype=np.float64)
        columns = self.history.columns
        self.history = pd.DataFrame(data=updatedData, index=updatedIndex, columns=columns)
        return

    def loadHistory(self, startTime_timestamp, endTime_timestamp):
        # makes sure that the length of the longest history is decisive
        start=time.time()
        data,index = self.api.returnChartData(pair=self.pairs[0], start=startTime_timestamp,
                                                end=endTime_timestamp, period=self.period)
        columns = [self.pairs[0]+label for label in ['_open', '_high', '_low', '_close', '_volume']]
        self.history=pd.DataFrame(data=data,index=index,columns=columns)
        for pair in self.pairs[1:]:
            data,index = self.api.returnChartData(pair=pair, start=startTime_timestamp,
                                              end=endTime_timestamp, period=self.period)
            columns = [pair + label for label in ['_open', '_high', '_low', '_close', '_volume']]
            tempDf=pd.DataFrame(data=data,index=index,columns=columns)
            #s = tempDf[pair+'_close'].isna().groupby(tempDf[pair+'_close'].notna().cumsum()).sum().max()

            consecutiveNaNs = self.maxConsecutivesNaNs(tempDf)
            maxRatio = self.maxRatioNaNs(tempDf)
            if consecutiveNaNs < 20 and maxRatio < 0.5:
                for timestamp in self.history.index:
                    #print(type(timestamp), timestamp)
                    dfIndex = np.array(tempDf.index)
                    dfIndex[np.all((timestamp <= dfIndex, dfIndex < timestamp + self.period), axis=0)] = timestamp
                    tempDf.index = list(dfIndex)
                self.history = pd.merge(self.history, tempDf, how='left', left_index=True, right_index=True)
            else:
                self.checker.mark(pair)
        self.checker.kickOut()
        end=time.time()
        print(f'preloading history tool {end-start} seconds')
        #print(self.history)
        return

    def getCurrentPrice(self, pair):
        lastPairPrice = self.api.getCurrentPrice(pair)
        return lastPairPrice

    def getCurrentTicker(self):
        currentTicker = self.api.returnTicker()
        return currentTicker
