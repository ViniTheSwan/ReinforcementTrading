import numpy as np
import pandas as pd
import math
import requests
import time
import os
import sys
from datetime import datetime
#technical analysis
import btalib
#preporcessing
from sklearn.preprocessing import MinMaxScaler
#tensorflow
import tensorflow as tf
from tensorflow.keras import Model,Sequential
from tensorflow.keras.layers import Dense,LSTM,BatchNormalization,Dropout,Input
from tensorflow.keras.models import load_model
##plotly
from plotly.offline import plot
import plotly.graph_objs as go
from plotly.subplots import make_subplots

#tensorflow setup
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

#defining constants
#labels must be included for
os.chdir('data')
columns=['close','high','low','volume','rising']
importCols=['labels','close','high','low','volume','rising']
indicators=['mfi']
maxLenDf= 500
#defining variables
period=300
pairs=['BTCUSDT','DASHBTC', 'DOGEBTC', 'LTCBTC', 'XEMBTC', 'XRPBTC','ETHBTC', 'SCBTC', 'DCRBTC', 'LSKBTC',
                    'STEEMBTC', 'ETCBTC', 'REPBTC', 'ARDRBTC', 'ZECBTC', 'STRATBTC', 'GNTBTC', 'ZRXBTC', 'CVCBTC', 'OMGBTC',
                    'STORJBTC', 'EOSBTC', 'SNTBTC', 'KNCBTC', 'BATBTC', 'LOOMBTC', 'QTUMBTC', 'MANABTC', 'BNTBTC', 'POLYBTC',
                    'ATOMBTC', 'TRXBTC', 'LINKBTC', 'XTZBTC', 'PAXBTC', 'SNXBTC', 'MATICBTC', 'MKRBTC', 'DAIBTC', 'NEOBTC',
                    'AVABTC', 'CHRBTC', 'BNBBTC', 'MDTBTC', 'LENDBTC', 'RENBTC', 'LRCBTC', 'WRXBTC', 'SXPBTC', 'STPTBTC']
selectedPairs=['BTCUSDT','DASHBTC', 'LTCBTC', 'XEMBTC', 'XRPBTC','ETHBTC', 'SCBTC', 'DCRBTC', 'LSKBTC',
                    'STEEMBTC', 'ETCBTC', 'REPBTC', 'ARDRBTC', 'ZECBTC', 'STRATBTC', 'GNTBTC', 'ZRXBTC', 'CVCBTC', 'OMGBTC',
                    'STORJBTC', 'EOSBTC', 'SNTBTC', 'KNCBTC', 'BATBTC', 'LOOMBTC', 'QTUMBTC', 'MANABTC', 'BNTBTC', 'POLYBTC',
                    'ATOMBTC', 'TRXBTC', 'LINKBTC', 'MATICBTC', 'NEOBTC',
                    'BNBBTC', 'LENDBTC', 'RENBTC', 'LRCBTC']
bigFive=['BTCUSDT','ETHBTC','BNBBTC','LTCBTC']
bigFive=['BTCUSDT','ETHBTC','BNBBTC']
period_dict = {60: "1m", 180: "3m", 300: "5m", 900: "15m", 1800: "30m", 3600: "1h", \
                            7200: "2h", 14400: "4h", 21600: "6h", 28800: "8h", 43200: "12h", 86400: "1d", \
                            259200: "3d", 1209600: "1w"}


def labeling(df,inputPairs=selectedPairs):
    if binary:
        for pair in inputPairs:
            df[pair+'_rising'] = (df[pair+'_close'].shift(1) < df[pair+'_close']).astype(np.int)
            print(df[pair+'_rising'].mean())
            #print(df[[pair+'_close',pair+'_rising']])
    else:
        for pair in inputPairs:
            df[pair+'_rising']=df[pair+'_close'].pct_change()
    return df

def chartToCsv( pair, start, end, period):
    print(pair)
    # print("here")
    start=start.timestamp()
    end=end.timestamp()
    starts = []
    timestamps = []
    chartLis = []
    period_str = period_dict[period]
    dfExists = False
    # the pairs are written in poloniex convention with quote_base and therefore have to be reversed
    # binance works with timestamps in miliseconds so our timestamps have to be converted
    start = 1000 * start
    end = 1000 * end
    # split the request in chunks that have in maximum 1000 datapoints
    numParts = math.ceil((end - start) / (period * 1e6))
    labels = ['open', 'high', 'low', 'close', 'volume', 'quoteAssetVolume', 'numberOfTrades',
              'takerBuyBaseAssetVolume', 'takerBuyQuoteAssetVolume']
    for i in range(len(labels)):
        labels[i]=pair + '_' +labels[i]
    for i in range(numParts):
        subStart = start + i * (end - start) / numParts
        subEnd = start + (i + 1) * (end - start) / numParts
        #print('start:', start, 'end:', end, 'subStart:', subStart, 'subEnd:', subEnd)
        url = 'https://api.binance.com/api/v1/klines?symbol={}&interval={}&startTime={}&endTime={}&limit=1000'.format( \
            pair, period_str, int(subStart), int(subEnd))
        for k in range(10):
            try:
                request = requests.get(url, timeout=1)
                request.raise_for_status()
                break
            except:
                print('request failed')
                time.sleep(1)
        chartLis += request.json()
    chartArray = np.float64(chartLis)
    timestamps = chartArray[:, 0] / 1000.0
    chartArray = np.delete(chartArray, np.array([0, 6, 11]), axis=1)
    addDf = pd.DataFrame(chartArray, columns=labels, index=timestamps)
    addDf.to_csv('historicalData{}{}.csv'.format(period, pair))

def combineDf():
    #pairsToPredict=pairs.copy()
    pairsToPredict=['BTCUSDT']
    dfAllExists=False
    pairCol=[]
    for pair in pairs:
        print(pair)
        maxLen=maxLenDf
        df = pd.read_csv(f'historicalData300{pair}.csv',index_col=0,header=0)
        df=df[[pair+'_'+col for col in columns]]
        #df.drop(df.index[0],axis=0,inplace=True)
        df.index.name=None
        df.columns.name=None
        df=df.sort_index()
        df=df.drop_duplicates(keep='first')
        if len(df.index)<maxLenDf:
            maxLen=len(df.index)-1
        #################maxlen################3
        #df = df.loc[df.index[-maxLen]:,:]
        pairCol.append(pair)
        if dfAllExists:
            #for timestamp in df.index:
                #dfIndex=np.array(df.index)
                #dfAllIndex=np.array(dfAll.index)
                #dfIndex[np.all((timestamp <= dfIndex, dfIndex < timestamp + period), axis=0)]= dfIndex[np.all((timestamp <= dfIndex, dfIndex < timestamp + period), axis=0)][0]
                #dfIndex[dfIndex==timestamp]=dfAllIndex[np.all((timestamp <= dfAllIndex, dfAllIndex < timestamp + period), axis=0)]
            for timestamp in dfAll.index:
                dfIndex=np.array(df.index)
                '''
                if len(dfIndex[np.all((timestamp <= dfIndex, dfIndex < timestamp + period), axis=0)]) > 1:
                    print(len(dfIndex[np.all((timestamp <= dfIndex, dfIndex < timestamp + period), axis=0)]))
                    dfIndex[np.all((timestamp <= dfIndex, dfIndex < timestamp + period), axis=0)][0]=timestamp
                    dfIndex[np.all((timestamp <= dfIndex, dfIndex < timestamp + period), axis=0)][1:]=timestamp+period
                else:
                    dfIndex[np.all((timestamp <= dfIndex, dfIndex < timestamp + period), axis=0)] = timestamp
                '''
                dfIndex[np.all((timestamp <= dfIndex, dfIndex < timestamp + period), axis=0)] = timestamp
            df.index=list(dfIndex)
            dfAll=pd.merge(dfAll,df,how='left',left_index=True,right_index=True)

        else:
            dfAll=df
            dfAllExists=True
        print('df index length:', len(df.index))
        print('dfAll index length:', len(dfAll.index))
    #dfAll.columns=pd.MultiIndex.from_product([pairCol,columns],names=['coins','attributes'])
    print(dfAll)
    dfAll=labeling(dfAll,binary=False,inputPairs=selectedPairs)
    print(dfAll)
    dfAll.to_csv(f'historicalData{period}All.csv')

#STEPS executed every time, the model gets called
def loadData(period=300):
    dfAll=pd.read_csv(f'historicalData{period}All.csv',index_col=0,header=0)
    return dfAll
def preProcess(df,predictPair=None,backWindow=500,futureWindow=1,split=0.8):
    scalerX=MinMaxScaler()
    scalerY=MinMaxScaler()
    YLis=[]
    pairLis=[]
    for pair in selectedPairs:
        YLis.append(pair+'_rising')
    for pair in selectedPairs:
        for col in columns:
            pairLis.append(pair+'_'+col)
    df=df[pairLis]
    df=df.loc[df.index[200000:],:]
    df.index=pd.to_datetime(df.index.astype('int'))
    df.index=pd.DatetimeIndex(df.index)
    df.interpolate(method='time',inplace=True)
    ######3df.dropna(inplace=True,axis=0)
    print('dfLen after dropping nas:',len(df.index))
    dfY=df[YLis]

    dfX=df.drop(YLis,axis=1)
    X=dfX.to_numpy()
    Y=dfY.to_numpy()
    if predictPair:
        Y=df[predictPair+'_rising']
    print('Y.shape:',Y.shape,'X.shape:',X.shape)
    lenTot=len(dfX.index)
    print(lenTot)
    XTrans=scalerX.fit_transform(X)
    if binary:
        YTrans=Y
    else:
        YTrans=scalerY.fit_transform(Y)
    indexTrain=df.index[:int(split*lenTot)]
    indexTest=df.index[int(split*lenTot):]
    XTrain=XTrans[:int(split*lenTot)]
    XTest=XTrans[int(split*lenTot):]
    YTrain=YTrans[:int(split*lenTot)]
    YTest=YTrans[int(split*lenTot):]
    index_train=[]
    index_test=[]
    X_train=[]
    X_test=[]
    Y_train=[]
    Y_test=[]
    for i in range(backWindow, XTrain.shape[0] - futureWindow):
        X_train.append(XTrain[i - backWindow:i + 1, :])
        Y_train.append(YTrain[i + futureWindow])
        index_train.append(indexTrain[i])
    for i in range(backWindow, XTest.shape[0] - futureWindow):
        X_test.append(XTest[i - backWindow:i + 1, :])
        Y_test.append(YTest[i + futureWindow])
        index_test.append(indexTest[i])
    X_train, Y_train, X_test, Y_test, index_train, index_test = np.array(X_train), np.array(Y_train), np.array(X_test), np.array(Y_test), np.array(index_train), np.array(index_test)
    print('X_train.shape:',X_train.shape,'Y_train.shape:',Y_train.shape)

    if binary:
        X_train,Y_train=balanceData(X_train,Y_train)

    if predictPair:
        print('reshaping')
        Y_train=Y_train.reshape(-1,1)
        Y_test=Y_test.reshape(-1,1)
    print('X_train.shape:', X_train.shape, 'Y_train.shape:', Y_train.shape)
    return X_train, Y_train, X_test, Y_test, index_train, index_test

def balanceData(X_train,Y_train):
    allInd = np.arange(0, len(X_train), 1)
    sells = allInd[Y_train == 0]
    buys = allInd[Y_train == 1]
    balanceLen = min(buys.shape[0], sells.shape[0])
    buys = buys[:balanceLen]
    sells = sells[:balanceLen]
    balancedIndex = np.append(buys, sells)
    print('balancedIndex:', balancedIndex)
    np.random.shuffle(balancedIndex)
    print('balancedIndex:', balancedIndex)
    X_train = X_train[balancedIndex, :, :]
    Y_train = Y_train[balancedIndex]
    print('newMean:', np.mean(Y_train))
    return X_train,Y_train

def buildModel(X_train,X_test,units=175):
    print((X_train.shape[1], X_train.shape[2]))
    model=Sequential()
    model.add(Input(shape=(X_train.shape[1], X_train.shape[2])))
    model.add(BatchNormalization())
    model.add(LSTM(units=units,activation= 'tanh',recurrent_activation='sigmoid',return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=units,activation= 'tanh',recurrent_activation='sigmoid',return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=units,activation='relu'))
    if binary:
        model.add(Dense(units=Y_train.shape[1]))
    else:
        model.add(Dense(units=Y_train.shape[1]))
    return model
def trainModel(model,X_train,Y_train):
    if binary:
        model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])
    else:
        model.compile(optimizer='Adam', loss='mean_squared_error', metrics=['accuracy'])
    print(model.summary())
    logdir='/home/christophboomer/Desktop/BotPandas/models/'
    if not os.path.exists(logdir):
        sys.exit(2)

    Name=f'LSTM{predictPair}'
    saveBest = tf.keras.callbacks.ModelCheckpoint(
       logdir+ Name + 'checkpoint.h5', monitor='val_loss', verbose=0, save_best_only=True,
        save_weights_only=False, mode='auto', save_freq='epoch', options=None
    )
    model.fit(X_train, Y_train, epochs=60, batch_size=60, validation_data=(X_test, Y_test),callbacks=[saveBest], shuffle=True)
    model.save(logdir+Name+'h5')

for pair in pairs:
    predictPair=pair
    selectedPairs=bigFive
    if pair not in selectedPairs:
        selectedPairs.append(pair)
    #combineDf()
    binary=True ###setting global variable
    dfAll=loadData()
    dfAll=labeling(dfAll)
    X_train, Y_train, X_test, Y_test, index_train, index_test= preProcess(dfAll,predictPair)
    model=buildModel(X_train,X_test)
    trainModel(model,X_train,Y_train)




#################plotting results#########################

#model=load_model('pctChangeModel1599696108.124432.h5')

Y_pred=model.predict(X_test)

#selecting only one coin

#Y_pred=Y_pred[:,ind]
#Y_test=Y_test[:,ind]
print(Y_pred)
#measured data
fig = make_subplots(rows=2, cols=1, shared_xaxes=True)
fig.add_trace(
    go.Scatter(
        x=index_test,
        y=Y_test,
        mode='lines',
        name='true',
        marker=dict(
            size=3,
            color='blue'
        )
    ),
    row=1, col=1
)
#predicted data
fig.add_trace(
    go.Scatter(
        x=index_test,
        y=Y_pred,
        mode='lines',
        name='predicted',
        marker=dict(
            size=3,
            color='red'
        )
    ),
    row=1, col=1
)

plot(fig)
