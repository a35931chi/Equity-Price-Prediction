import numpy as np
import pandas as pd
import time

from sklearn.preprocessing import MinMaxScaler

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Activation

import matplotlib.pyplot as plt

#load prices
df = pd.read_csv('Data/AAPL-yahoo.csv', index_col = 'Date')

#normalize
df['Norm Adj Close'] = (df['Adj Close'] - df['Adj Close'].mean()) / (df['Adj Close'].max() - df['Adj Close'].min())
df['Norm Vol'] = (df['Volume'] - df['Volume'].mean()) / (df['Volume'].max() - df['Volume'].min())

#should compare normalization results with scaler
scaler = MinMaxScaler(feature_range=(0, 1))
yahoo_stock_prices = scaler.fit_transform(df)

#print(df.head())


data_adjclose = np.array(df['Norm Adj Close'])
data_adjvol = np.array(df['Norm Vol'])

if False: #output normalized plots
    plt.figure(figsize = (10, 5))
    plt.plot(data_adjclose)
    plt.title('Adj Close')
    plt.xlabel('time period')
    plt.ylabel('normalize series value')
    plt.tight_layout()
    plt.show()

    plt.figure(figsize = (10, 5))
    plt.plot(data_adjvol)
    plt.title('Volume')
    plt.xlabel('time period')
    plt.ylabel('normalize series value')
    plt.tight_layout()
    plt.show()


def window_transform_series(series, window_size):
    # containers for input/output pairs
    X = []
    y = []
    #print(series)
    #print(window_size)
    for i in range(len(series) - window_size):
        X.append(series[i: i + window_size])
        y.append(series[i + window_size])
        #print(i)
        #print(series[i: i + window_size])
        #print(series[i + window_size])
    # reshape each
    print(type(X))
    X = np.asarray(X)
    print(type(X))
    print(np.shape(X)[0:2])
    X.shape = (np.shape(X)[0:2])
    print(X.shape)
    what = input('what')
    y = np.asarray(y)
    y.shape = (len(y),1)

    return X,y

def window_transform_series_v2(series, window_size, days_out):
    # y is actually quite a few days into the future
    X = []
    y = []
    #print(series)
    #print(window_size)
    for i in range(len(series) - window_size - days_out):
        X.append(series[i: i + window_size])
        y.append(series[i + window_size + days_out])
        print(i)
        print(series[i: i + window_size])
        print(series[i + window_size + days_out])
    # reshape each 
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:2])
    y = np.asarray(y)
    y.shape = (len(y),1)

    return X,y

if True:
    ind = []
    training_error = []
    testing_error = []
    model_fit_time = []
    training_predict_time = []
    testing_predict_time = []
    training_error_eval_time = []
    testing_error_eval_time = []
    window_sizes = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    epochs = [10, 100, 250, 500, 1000, 2000]
    batch_sizes = [10, 50, 100, 250, 500]
    for window_size in window_sizes:
        for epoch in epochs:
            for batch_size in batch_sizes:
                X, y = window_transform_series(series = data_adjclose,
                                               window_size = window_size)
                print(type(X), type(y))
                print(X.shape, y.shape)
                what = input('what')

                # X.shape: (4576,7) and y.shape: (4576, 1)
                # split our dataset into training / testing sets
                train_test_split = int(np.ceil(2 * len(y) / float(3)))   # set the split point: 3051

                X_train, y_train = X[:train_test_split, :], y[:train_test_split] #X_train.shape: (3051, 7), y_train: (3051, 1)
                X_test, y_test = X[train_test_split:, :], y[train_test_split:] #x_test.shape: (1525, 7), y_test: (1525, 1)


                # NOTE: to use keras's RNN LSTM module our input must be reshaped to [samples, window size, stepsize] 
                X_train = np.asarray(np.reshape(X_train, (X_train.shape[0], window_size, 1))) #(3051, 7, 1)
                X_test = np.asarray(np.reshape(X_test, (X_test.shape[0], window_size, 1))) #(1525, 7, 1)


                model = Sequential()
                model.add(LSTM(7, input_shape = (window_size, 1)))
                model.add(Dense(1))

                optimizer = keras.optimizers.RMSprop(lr = 0.001, rho = 0.9, epsilon = 1e-08, decay = 0.0)
                model.compile(loss = 'mean_squared_error', optimizer = optimizer)
                #print(model.summary())

                t0 = time.time()
                model.fit(X_train, y_train, epochs = epoch, batch_size = batch_size, verbose = 0)
                t1 = time.time()

                train_predict = model.predict(X_train)
                t2 = time.time()
                test_predict = model.predict(X_test)
                t3 = time.time()

                # evaluate training and testing errors
                training_error = model.evaluate(X_train, y_train, verbose=0)
                t4 = time.time()
                testing_error = model.evaluate(X_test, y_test, verbose=0)
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

                ind.append((window_size, epoch, batch_size))
                training_error.append(training_error)
                testing_error.append(testing_error)
                model_fit_time.append(t1 - t0)
                training_predict_time.append(t2 - t1)
                testing_predict_time.append(t3 - t2)
                training_error_eval_time.append(t4 - t3)
                testing_error_eval_time.append(t5 - t4)

if False: #plot results
    # plot original series
    plt.figure(figsize = (10, 5))
    plt.plot(data_adjclose, color = 'k', alpha = 0.5)

    # plot training set prediction
    split_pt = train_test_split + window_size 
    plt.plot(np.arange(window_size , split_pt, 1), train_predict, color = 'b')

    # plot testing set prediction
    plt.plot(np.arange(split_pt, split_pt + len(test_predict), 1),
             test_predict, color = 'r')

    # pretty up graph
    plt.xlabel('day')
    plt.ylabel('(normalized) price of Apple stock')
    plt.legend(['original series', 'training fit', 'testing fit'])
    plt.tight_layout()
    plt.show()

if False:
    # let's try multiple layers
    ind = []
    training_error = []
    testing_error = []
    model_fit_time = []
    training_predict_time = []
    testing_predict_time = []
    training_error_eval_time = []
    testing_error_eval_time = []
    window_sizes = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    epochs = [10, 100, 250, 500, 1000, 2000]
    batch_sizes = [10, 50, 100, 250, 500]
    for window_size in window_sizes:
        for epoch in epochs:
            for batch_size in batch_sizes:
                X, y = window_transform_series(series = data_adjclose,
                                               window_size = window_size)

                # X.shape: (4576,7) and y.shape: (4576, 1)
                # split our dataset into training / testing sets
                train_test_split = int(np.ceil(2 * len(y) / float(3)))   # set the split point: 3051

                X_train, y_train = X[:train_test_split, :], y[:train_test_split] #X_train.shape: (3051, 7), y_train: (3051, 1)
                X_test, y_test = X[train_test_split:, :], y[train_test_split:] #x_test.shape: (1525, 7), y_test: (1525, 1)


                # NOTE: to use keras's RNN LSTM module our input must be reshaped to [samples, window size, stepsize] 
                X_train = np.asarray(np.reshape(X_train, (X_train.shape[0], window_size, 1))) #(3051, 7, 1)
                X_test = np.asarray(np.reshape(X_test, (X_test.shape[0], window_size, 1))) #(1525, 7, 1)


                model = Sequential()
                model.add(LSTM(7, input_shape = (window_size, 1)))
                model.add(Dense(1))

                optimizer = keras.optimizers.RMSprop(lr = 0.001, rho = 0.9, epsilon = 1e-08, decay = 0.0)
                model.compile(loss = 'mean_squared_error', optimizer = optimizer)
                #print(model.summary())

                t0 = time.time()
                model.fit(X_train, y_train, epochs = epoch, batch_size = batch_size, verbose = 0)
                t1 = time.time()

                train_predict = model.predict(X_train)
                t2 = time.time()
                test_predict = model.predict(X_test)
                t3 = time.time()

                # evaluate training and testing errors
                training_error = model.evaluate(X_train, y_train, verbose=0)
                t4 = time.time()
                testing_error = model.evaluate(X_test, y_test, verbose=0)
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

                ind.append((window_size, epoch, batch_size))
                training_error.append(training_error)
                testing_error.append(testing_error)
                model_fit_time.append(t1 - t0)
                training_predict_time.append(t2 - t1)
                testing_predict_time.append(t3 - t2)
                training_error_eval_time.append(t4 - t3)
                testing_error_eval_time.append(t5 - t4)
