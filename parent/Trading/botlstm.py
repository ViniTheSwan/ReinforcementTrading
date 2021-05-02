# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 19:35:18 2020

@author: christophboomer
"""
# todo: christophboomer@pop-os:~/PycharmProjects/Bot$ tensorboard --logdir=logs/ to get the url of tensorboard

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
import tensorflow as tf
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import LSTM,Dropout,BatchNormalization, Input,Dense,Flatten
from tensorflow.keras.callbacks import EarlyStopping,TensorBoard
from tensorflow.keras.models import load_model
from tensorflow.keras import regularizers
from sklearn.preprocessing import MinMaxScaler

##plotly
from plotly.offline import plot
import plotly.graph_objs as go
from plotly.subplots import make_subplots

#btalib
#import btalib


physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

# todo: @Vincent check out the possibility of running multiple models (incl. training) the same time
# run multiple models the same time with a single gpu using the code below
# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
# sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
df = pd.read_csv('historicalData300BNBBTCtrade.csv', index_col=0, header=0)
print(df.columns)
print(df['trading'].mean())
df.dropna(inplace=True,axis=0)
#df=df[['open','high','low','close','volume','mfi','sma','wma','mom','cci']]
df=df[['close','high','low','volume','mfi','trading']]
'''
for col in df.columns:
    if col != 'trading':
        df[col]=df[col].pct_change()
'''
df.dropna(inplace=True,axis=0)

Y=df['trading'].to_numpy()
df=df.drop(labels='trading',axis=1)
X=df.to_numpy()
index = df.index
scaler = MinMaxScaler()
XTrans = scaler.fit_transform(X)
XTrain = XTrans[:int(XTrans.shape[0] * 0.7), :]
XTest = XTrans[int(XTrans.shape[0] * 0.7):, :]
YTrain=Y[:int(XTrans.shape[0]*0.7)]
YTest=Y[int(XTrans.shape[0]*0.7):]
indexTrain = df.index[:int(XTrans.shape[0] * 0.7)]
indexTest = df.index[int(XTrans.shape[0] * 0.7):]

X_train = []
Y_train = []
X_test = []
Y_test = []
futureWindow = 0
backWindow = 700
RNN_units=120

index_train=[]
index_test=[]

for i in range(backWindow, XTrain.shape[0] - futureWindow):
    X_train.append(XTrain[i - backWindow:i+1, :])
    Y_train.append(YTrain[i + futureWindow])
    index_train.append(indexTrain[i])
for i in range(backWindow, XTest.shape[0] - futureWindow):
    X_test.append(XTest[i - backWindow:i+1, :])
    Y_test.append(YTest[i + futureWindow])
    index_test.append(indexTest[i])

X_train, Y_train, X_test, Y_test, index_train,index_test = np.array(X_train), np.array(Y_train), np.array(X_test), np.array(Y_test), np.array(index_train), np.array(index_test)

all=np.arange(0,len(X_train),1)

sells=all[Y_train==0]
buys=all[Y_train==1]
balanceLen=min(buys.shape[0],sells.shape[0])
buys=buys[:balanceLen]
sells=sells[:balanceLen]
balancedIndex=np.append(buys,sells)
print('balancedIndex:',balancedIndex)
np.random.shuffle(balancedIndex)
print('balancedIndex:',balancedIndex)
X_train=X_train[balancedIndex,:,:]

Y_train=Y_train[balancedIndex]

print('newMean:',np.mean(Y_train))


print("Ytrain shape:",Y_train.shape)
print("Ytest shape:",Y_test.shape)

print('Xtest', X_test.shape, 'Xtrain', X_train.shape)
print('Y_test', Y_test.shape, 'Y_train', Y_train.shape)

activation='tanh'
recurrent_activation='sigmoid'


def buildModel():
    model = Sequential()
    print((X_train.shape[1], X_train.shape[2]))
    model.add(Input(shape=(X_train.shape[1], X_train.shape[2])))
    model.add(BatchNormalization())
    model.add(LSTM(units=RNN_units, activation=activation,
                   recurrent_activation=recurrent_activation, use_bias=True,
                   kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal',
                   bias_initializer='zeros', unit_forget_bias=True, kernel_regularizer=regularizers.l1_l2(0.0001,0.0001),
                   recurrent_regularizer=regularizers.l1_l2(0.0001,0.0001), bias_regularizer=regularizers.l1_l2(0.0001,0.0001), activity_regularizer=regularizers.l1_l2(0.0001,0.0001),
                   kernel_constraint=None, recurrent_constraint=None, bias_constraint=None,
                   dropout=0.0, recurrent_dropout=0, implementation=2, return_sequences=True,
                   return_state=False, go_backwards=False, stateful=False, time_major=False,
                   unroll=False))
    model.add(Dropout(0.2))
    model.add(LSTM(input_shape=(X_train.shape[1], X_train.shape[2]), units=RNN_units, activation=activation,
                   recurrent_activation=recurrent_activation, use_bias=True,
                   kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal',
                   bias_initializer='zeros', unit_forget_bias=True, kernel_regularizer=regularizers.l1_l2(0.0001,0.0001),
                   recurrent_regularizer=regularizers.l1_l2(0.0001,0.0001), bias_regularizer=regularizers.l1_l2(0.0001,0.0001), activity_regularizer=regularizers.l1_l2(0.0001,0.0001),
                   kernel_constraint=None, recurrent_constraint=None, bias_constraint=None,
                   dropout=0.0, recurrent_dropout=0, implementation=2, return_sequences=True,
                   return_state=False, go_backwards=False, stateful=False, time_major=False,
                   unroll=False))
    model.add(Dropout(0.2))
    model.add(LSTM(input_shape=(X_train.shape[1], X_train.shape[2]), units=RNN_units, activation=activation,
                   recurrent_activation=recurrent_activation, use_bias=True,
                   kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal',
                   bias_initializer='zeros', unit_forget_bias=True, kernel_regularizer=regularizers.l1_l2(0.0001,0.0001),
                   recurrent_regularizer=regularizers.l1_l2(0.0001,0.0001), bias_regularizer=regularizers.l1_l2(0.0001,0.0001), activity_regularizer=regularizers.l1_l2(0.0001,0.0001),
                   kernel_constraint=None, recurrent_constraint=None, bias_constraint=None,
                   dropout=0.0, recurrent_dropout=0, implementation=2, return_sequences=True,
                   return_state=False, go_backwards=False, stateful=False, time_major=False,
                   unroll=False))
    model.add(Dropout(0.2))
    model.add(LSTM(input_shape=(X_train.shape[1], X_train.shape[2]), units=RNN_units, activation=activation,
                   recurrent_activation=recurrent_activation, use_bias=True,
                   kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal',
                   bias_initializer='zeros', unit_forget_bias=True, kernel_regularizer=regularizers.l1_l2(0.0001,0.0001),
                   recurrent_regularizer=regularizers.l1_l2(0.0001,0.0001), bias_regularizer=regularizers.l1_l2(0.0001,0.0001), activity_regularizer=regularizers.l1_l2(0.0001,0.0001),
                   kernel_constraint=None, recurrent_constraint=None, bias_constraint=None,
                   dropout=0.0, recurrent_dropout=0, implementation=2, return_sequences=False,
                   return_state=False, go_backwards=False, stateful=False, time_major=False,
                   unroll=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=RNN_units,activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(units=1,activation='sigmoid'))
    '''
    input=Input(shape=(X_train.shape[1], X_train.shape[2]))
    f1=Flatten(input)
    l1=Dense(units=RNN_units,activation='relu')(f1)
    d1=Dropout(0.2)(l1)
    l2=Dense(units=RNN_units,activation='relu')(d1)
    d2 = Dropout(0.2)(l2)
    l3 = Dense(units=RNN_units, activation='relu')(d2)
    d3 = Dropout(0.2)(l3)
    l4 = Dense(units=RNN_units, activation='relu')(d3)
    d4 = Dropout(0.2)(l4)
    output=Dense(units=1,activation='sigmoid')(d4)
    model=Model(inputs = input, outputs=output)
    print(model.summary())
    '''

    '''
    #dense model
    model.add(Input(shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dense(units=RNN_units,activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(units=RNN_units*2, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(units=RNN_units * 3, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(units=RNN_units * 3, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(units=RNN_units*2, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(units=RNN_units*1, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(units=RNN_units * 0.5, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(units=1,activation='sigmoid'))
    '''
    model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])
    print(model.summary())
    return model


def trainModel(model):
    numLayers=len(model.layers)
    Name="LSTM_act:{}_recact:{}_numlayers:{}_time:{}".format(activation, recurrent_activation, numLayers,time.time())
    saveBest = tf.keras.callbacks.ModelCheckpoint(
     Name+'checkpoint',   monitor='val_loss', verbose=0, save_best_only=True,
        save_weights_only=False, mode='auto', save_freq='epoch', options=None
    )
    tensorboard = TensorBoard(log_dir='logs/{}'.format(Name))
    model.fit(X_train, Y_train, epochs=50, batch_size=60, validation_data=(X_test, Y_test), callbacks=[tensorboard,saveBest], shuffle=True)


def saveModel(model, name):
    model.save(f'{name}{time.time()}.h5')



#model=buildModel()
#trainModel(model)
#saveModel(model,'newModel')



model=load_model('LSTM_act:tanh_recact:sigmoid_numlayers:12_time:1599091577.9448771checkpoint')

######testing######
df = pd.read_csv('historicalData300BTCUSDT.csv', index_col=0, header=1)
df=df[['close','high','low','volume']]
df['mfi']=btalib.mfi(df).df
df.dropna(inplace=True,axis=0)
X=df.to_numpy()
XTrans=scaler.transform(X)
XTrain = XTrans[:int(XTrans.shape[0] * 0.7), :]
XTest = XTrans[int(XTrans.shape[0] * 0.7):, :]
indexTest = df.index[int(XTrans.shape[0] * 0.7):]
X_test=[]
index_test=[]
for i in range(backWindow, XTest.shape[0] - futureWindow):
    X_test.append(XTest[i - backWindow:i+1, :])
    index_test.append(indexTest[i])
X_test,index_test=np.array(X_test),np.array(index_test)
##############

Y_pred=np.ravel(model.predict(X_test[:,:,:]))
print(Y_pred.shape)



fig = make_subplots(rows=2, cols=1, shared_xaxes=True)
datY = X_test[:,-1,2]
# trace_open
fig.add_trace(
    go.Scatter(
        x=index_test[Y_pred>=0.5],
        y=datY[Y_pred>=0.5],
        mode='markers',
        name='close',
        marker=dict(
            size=4,
            color='red'
        )
    ),
    row=1, col=1
)
#trace_close
fig.add_trace(
    go.Scatter(
        x=index_test[Y_pred<0.5],
        y=datY[Y_pred<0.5],
        mode='markers',
        name='close',
        marker=dict(
            size=4,
            color='green'
        )
    ),
    row=1, col=1
)

fig.add_trace(
    go.Scatter(
        x=index_test,
        y=datY,
        mode='lines',
        name='close',
        marker=dict(
            size=3,
            color='black'
        )
    ),
    row=1, col=1
)

plot(fig)


#model = buildModel()
#trainModel(model)
#saveModel(model,'newModel')

