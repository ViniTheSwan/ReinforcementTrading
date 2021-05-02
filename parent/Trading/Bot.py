import sys
import getopt
import time
from datetime import datetime

from BotChart import BotChart
from BotStrategy import BotStrategy
#from botcandlestick import BotCandlestick
from BotAPI import BotAPI
import numpy as np
from BotChecker import BotChecker
import pandas as pd

import math


def main(argv):
    # default:
    period = 300
    backTest = False
    trading = True
    csvName = "defaultLog.csv"

    try:
        # getopt.getopt returns two elements, opts which consists of pairs (options and values)
        # the second (args) is the list of program arguments left after the option list was stripped
        # (this is a trailing slice of args)
        opts, args = getopt.getopt(argv, "p:l:s:e:b", ["period=","startTime=",
                                                       "endTime=","liveTest",
                                                        "backTest", "animation", "help"])
    except getopt.GetoptError:
        print("Getopt Error")
        sys.exit(2)

    for opt, arg in opts:
        print(opt, arg)

        if opt in ("-b", "--backTest"):
            backTest = True
            trading = False
        elif opt in ("-s", "--startTime"):
            startTime_datetime = datetime.strptime(arg, '%Y-%m-%d')
            startTime_timestamp = datetime.timestamp(startTime_datetime)
        elif opt in ("-e", "--endTime"):
            endTime_datetime = datetime.strptime(arg, '%Y-%m-%d')
            endTime_timestamp = datetime.timestamp(endTime_datetime)
        elif opt in ('-l','--liveTest'):
            print('not trading')
            trading=False
        elif opt in ("-p", "--period"):

            if arg in ["60", "180", "300", "900", "1800", "3600", "7200", "14400", "21600", "28800", "43200", "86400",
                       "259200", "1209600"]:
                period = int(arg)
            elif arg in ["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d", "3d", "1w", "1M"]:
                period_dict = {"1m": 60, "3m": 180, "5m": 300, "15m": 900, "30m": 1800, "1h": 3600, \
                               "2h": 7200, "4h": 14400, "6h": 21600, "8h": 28800, "12h": 43200, "1d": 86400, \
                               "3d": 259200, "1w": 1209600}
                period = period_dict[arg]
            else:
                print("Binance requires periods in 60,180,300, 900, 1800, 3600, 7200, 14400, 21600,28800,43200,86400,259200,1209600 or\
                  1m,3m,5m,15m,30m,1h,2h,4h,6h,8h,12h,1d,3d,1w,1M")
                sys.exit(2)

    print(period)

    checker = BotChecker()
    pairs=checker.pairs
    api = BotAPI(period = period)
    chart = BotChart(period=period)
    strategy = BotStrategy( period=period, startBalance=0.01,trading=trading)
    #determine the longest data requrement
    maxTime = 3000 #TODO determine Maxtime
    if backTest:
        chart.loadHistory(startTime_timestamp=startTime_timestamp,
                          endTime_timestamp=endTime_timestamp)
        for pos in range(maxTime//period,len(chart.history.index),):
            strategy.evaluatePosition(pos=pos)
            print("{} days passed".format((pos*period) / 86400))


    else:
        intermediateStep = time.time()
        chart.loadHistory(startTime_timestamp=intermediateStep - maxTime, endTime_timestamp=intermediateStep)
        ################ sync to binance ######################
        now = time.time()
        stepSize = period
        timestamp = stepSize * math.floor(now / stepSize) + stepSize
        if chart.history.index[-1]+period < timestamp:
            allData,index=api.returnChartData(pairs[0],start=intermediateStep,end=time.time(),period=period)
            for pair in pairs[1:]:
                data,index=api.returnChartData(pair,start=intermediateStep,end=time.time(),period=period)
                for i in range(len(allData)):
                    allData[i].extend(data[i])
            chart.appendData(allData,index)
        time.sleep(timestamp - now + 10)
        ############################
        while True:
            print('#########################################################NEW_TICK###########################################################################')
            startTime = time.time()  # ensure, that loop is exactly 5 minutes long
            chartLis, index = chart.getCurrentTicker()
            chart.appendData(chartLis,index)
            pos=-1
            strategy.evaluatePosition(pos=pos)
            endTime = time.time()
            time.sleep(period - (endTime - startTime))


if __name__ == "__main__":
    main(sys.argv[1:])
