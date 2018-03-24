import numpy as np
import pandas as pd

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Activation

import matplotlib.pyplot as plt

#load prices
df = pd.read_csv('Data/AAPL.csv', index_col = 'Date')

#normalize
df['Norm Adj Close'] = (df['Adj Close'] - df['Adj Close'].mean()) / (df['Adj Close'].max() - df['Adj Close'].min())
df['Norm Vol'] = (df['Volume'] - df['Volume'].mean()) / (df['Volume'].max() - df['Volume'].min())
print(df.head())

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
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:2])
    y = np.asarray(y)
    y.shape = (len(y),1)

    return X,y

window_size = 7
X, y = window_transform_series(series = data_adjclose,
                               window_size = window_size)

# X.shape: (4576,7) and y.shape: (4576, 1)
# split our dataset into training / testing sets
train_test_split = int(np.ceil(2 * len(y) / float(3)))   # set the split point: 3051

X_train, y_train = X[:train_test_split, :], y[:train_test_split] #X_train.shape: (3051, 7), y_train: (3051, 1)
X_test, y_test = X[train_test_split:, :], y[train_test_split:] #x_test.shape: (1525, 7), y_test: (1525, 1)

