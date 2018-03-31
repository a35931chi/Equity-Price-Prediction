import numpy as np
import pandas as pd
import time

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Activation

import matplotlib.pyplot as plt
import seaborn as sns

from Extract_Data import extract_data

pca, PCA_df, df = extract_data('AAPL')
PCA_df = shuffle(PCA_df, random_state = 0)
df = shuffle(df, random_state = 0)

PCA_X = PCA_df[['Dimension 1', 'Dimension 2', 'Dimension 3', 'Dimension 4',
                   'Dimension 5', 'Dimension 6', 'Dimension 7']]
PCA_y = PCA_df[['Adj Close 1day', 'Adj Close 5day',
                'Adj Close 1day pct_change', 'Adj Close 5day pct_change',
                'Adj Close 1day pct_change cls', 'Adj Close 5day pct_change cls']]

#split into train, test, validation sets
PCA_Xtrain, PCA_Xtest, PCA_ytrain, PCA_ytest = train_test_split(PCA_X, PCA_y, test_size = 0.2,
                                                                random_state = 42)

#try benchmark models:
#1. 
