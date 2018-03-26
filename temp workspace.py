import numpy as np
import pandas as pd
import time

from sklearn.preprocessing import MinMaxScaler

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Activation
from keras.layers import Dropout

import matplotlib.pyplot as plt

#load prices
df_aapl = pd.read_csv('Data/AAPL.csv', index_col = 'Date')

#normalize
scaler = MinMaxScaler(feature_range=(0, 1))
np_aapl = scaler.fit_transform(df_aapl)

df_aapl['Norm Adj Close'] = (df_aapl['Adj Close'] - df_aapl['Adj Close'].mean()) / (df_aapl['Adj Close'].max() - df_aapl['Adj Close'].min())
df_aapl['Norm Vol'] = (df_aapl['Volume'] - df_aapl['Volume'].mean()) / (df_aapl['Volume'].max() - df_aapl['Volume'].min())

if False:
    plt.plot(df_aapl[['Norm Adj Close']], label = 'calculated')
    plt.plot(np_aapl[:,4], label = 'scaler')
    plt.legend()
    plt.show()

def window_transform_series_v2(series, window_size, days_out):
    # y is actually quite a few days into the future
    X = []
    y = []
    #print(series)
    #print(window_size)
    days_out = days_out - 1
    for i in range(len(series) - window_size - days_out):
        X.append(series[i: i + window_size])
        y.append(series[i + window_size + days_out])
        #print(i)
        #print(series[i: i + window_size])
        #print(series[i + window_size + days_out])
    # reshape each 
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:2])
    y = np.asarray(y)
    y.shape = (len(y),1)

    return X, y

window_size = 5
days_out = 5
epoch = 500
batch_size = 125
X, y = window_transform_series_v2(np_aapl[:,4], window_size, days_out)
print(X[:5])
print(y[:5])
print(np_aapl[:20, 4])


train_size = int(len(X) * 0.80)
test_size = len(X) - train_size
Xtrain, Xtest = X[0:train_size, :], X[train_size:len(X), :]
ytrain, ytest = y[0:train_size, :], y[train_size:len(X), :]


Xtrain = np.asarray(np.reshape(Xtrain, (Xtrain.shape[0], window_size, 1))) #
Xtest = np.asarray(np.reshape(Xtest, (Xtest.shape[0], window_size, 1))) #

print(Xtrain.shape, Xtest.shape)

model = Sequential()

model.add(LSTM(input_dim = 1, output_dim = 50, return_sequences = True))
model.add(Dropout(0.2))

model.add(LSTM(100, return_sequences = False))
model.add(Dropout(0.2))

model.add(Dense(output_dim = 1))
model.add(Activation('linear'))

model.compile(loss = 'mse', optimizer = 'rmsprop')

t0 = time.time()
model.fit(Xtrain, ytrain, batch_size = batch_size, epoch = epoch,
          validation_split = 0.05, verbose = 0)
t1 = time.time()

train_predict = model.predict(Xtrain)
t2 = time.time()
test_predict = model.predict(Xtest)
t3 = time.time()

# evaluate training and testing errors
training_error = model.evaluate(Xtrain, ytrain, verbose = 0)
t4 = time.time()
testing_error = model.evaluate(Xtest, ytest, verbose = 0)
t5 = time.time()

print('\n')
print('window: {}, epoch: {}, batch size: {}'.format(window_size, epoch, batch_size))
print('training error = ' + str(training_error))
print('testing error = ' + str(testing_error))
print('model fit time: {0:.5g} s'.format(t1 - t0))
print('training predict time: {0:.5g} s'.format(t2 - t1))
print('testing predict time: {0:.5g} s'.format(t3 - t2))
print('training error eval time: {0:.5g} s'.format(t4 - t3))
print('testing error eval time: {0:.5g} s'.format(t5 - t4))
print ('compilation time : ', time.time() - t0)
