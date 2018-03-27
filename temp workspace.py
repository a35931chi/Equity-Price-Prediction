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

if False: #comparing two normalization techniques
    plt.plot(df_aapl[['Norm Adj Close']], label = 'calculated')
    plt.plot(np_aapl[:,4], label = 'scaler')
    plt.legend()
    plt.show()

def window_transform_series_v2(series, window_size, days_out):
    # y is actually quite a few days into the future
    X = []
    y = []
    y2 = []
    #print(series)
    #print(window_size)
    days_out = days_out - 1
    for i in range(len(series) - window_size - days_out):
        X.append(series[i: i + window_size])
        y.append(series[i + window_size + days_out])
        diff = series[i + window_size + days_out] / series[i + window_size] - 1
        if diff > 0.05:
            y2.append(2)
        elif diff > 0.01:
            y2.append(1)
        elif diff > -0.01:
            y2.append(0)
        elif diff > -0.05:
            y2.append(-1)
        else:
            y2.append(-2)
        #print(i)
        #print(series[i: i + window_size])
        #print(series[i + window_size + days_out])
    # reshape each 
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:2])
    y = np.asarray(y)
    y2 = np.asarray(y2)
    y.shape = (len(y), 1)
    y2.shape = (len(y2), 1)

    return X, y, y2

window_size = 5
days_out = 5
epoch = 500
batch_size = 125
X, y, y2 = window_transform_series_v2(np_aapl[:,4], window_size, days_out)
#print(X[:5])
#print(y[:5])
#print(np_aapl[:20, 4])


train_size = int(len(X) * 0.80)
test_size = len(X) - train_size
Xtrain, Xtest = X[0:train_size, :], X[train_size:len(X), :]
ytrain, ytest = y[0:train_size, :], y[train_size:len(X), :]
y2train, y2test = y2[0:train_size, :], y2[train_size:len(X), :]


Xtrain = np.asarray(np.reshape(Xtrain, (Xtrain.shape[0], window_size, 1))) #
Xtest = np.asarray(np.reshape(Xtest, (Xtest.shape[0], window_size, 1))) #

#print(Xtrain.shape, Xtest.shape)

ind = []
training_errors = []
testing_errors = []
model_fit_time = []
training_predict_time = []
testing_predict_time = []
training_error_eval_time = []
testing_error_eval_time = []

for first in [10, 50, 100, 200]:
    for second in [10, 50, 100, 200]:
        for dropout in [True, False]:
            
            model = Sequential()

            model.add(LSTM(output_dim = first, input_shape = (window_size, 1),
                           return_sequences = True))
            if dropout:
                model.add(Dropout(0.2))

            model.add(LSTM(second, return_sequences = False))
            if dropout:
                model.add(Dropout(0.2))

            model.add(Dense(output_dim = 1))
            model.add(Activation('linear'))

            model.compile(loss = 'mse', optimizer = 'rmsprop')

            t0 = time.time()
            model.fit(Xtrain, ytrain, batch_size = batch_size, epochs = epoch,
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
            print(first, second, dropout)
            print('training error = ' + str(training_error))
            print('testing error = ' + str(testing_error))
            print('model fit time: {0:.5g} s'.format(t1 - t0))
            print('training predict time: {0:.5g} s'.format(t2 - t1))
            print('testing predict time: {0:.5g} s'.format(t3 - t2))
            print('training error eval time: {0:.5g} s'.format(t4 - t3))
            print('testing error eval time: {0:.5g} s'.format(t5 - t4))

            ind.append((window_size, epoch, batch_size, first, second, dropout))
            training_errors.append(training_error)
            testing_errors.append(testing_error)
            model_fit_time.append(t1 - t0)
            training_predict_time.append(t2 - t1)
            testing_predict_time.append(t3 - t2)
            training_error_eval_time.append(t4 - t3)
            testing_error_eval_time.append(t5 - t4)
