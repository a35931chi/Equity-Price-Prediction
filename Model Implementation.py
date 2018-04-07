import numpy as np
import pandas as pd
from time import gmtime, strftime, time

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, ShuffleSplit, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler

from sklearn.metrics import make_scorer, mean_squared_error, accuracy_score

from sklearn.linear_model import Lasso
import xgboost as xgb

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Activation
from keras.wrappers.scikit_learn import KerasRegressor


import matplotlib.pyplot as plt
import seaborn as sns

from Extract_Data import extract_data

pca, PCA_df, df = extract_data('AAPL')
PCA_df_rand = shuffle(PCA_df, random_state = 0)
df_rand = shuffle(df, random_state = 0)

PCA_rand_X = PCA_df_rand[['Dimension 1', 'Dimension 2', 'Dimension 3', 'Dimension 4',
                          'Dimension 5', 'Dimension 6', 'Dimension 7']]
PCA_rand_y = PCA_df_rand[['Adj Close 1day', 'Adj Close 5day',
                          'Adj Close 1day pct_change', 'Adj Close 5day pct_change',
                          'Adj Close 1day pct_change cls', 'Adj Close 5day pct_change cls']]

PCA_X = PCA_df[['Dimension 1', 'Dimension 2', 'Dimension 3', 'Dimension 4',
                'Dimension 5', 'Dimension 6', 'Dimension 7']]
PCA_y = PCA_df[['Adj Close 1day', 'Adj Close 5day',
                'Adj Close 1day pct_change', 'Adj Close 5day pct_change',
                'Adj Close 1day pct_change cls', 'Adj Close 5day pct_change cls']]

df_rand_X = df_rand[['Open', 'High', 'Low', 'Adj Close', 'Volume', 'Range', 'MA5 Adj Close', 'MA5 Volume',
                     'MA5 Adj Close pct_change', 'MA5 Volume pct_change']]
df_rand_y = df_rand[['Adj Close 1day', 'Adj Close 5day', 'Adj Close 1day pct_change',
                     'Adj Close 5day pct_change', 'Adj Close 1day pct_change cls',
                     'Adj Close 5day pct_change cls']]

df_X = df[['Open', 'High', 'Low', 'Adj Close', 'Volume', 'Range', 'MA5 Adj Close', 'MA5 Volume',
           'MA5 Adj Close pct_change', 'MA5 Volume pct_change']]
df_y = df[['Adj Close 1day', 'Adj Close 5day', 'Adj Close 1day pct_change',
           'Adj Close 5day pct_change', 'Adj Close 1day pct_change cls',
           'Adj Close 5day pct_change cls']]

#split into train, test, validation sets
PCA_rand_Xtrain, PCA_rand_Xtest, PCA_rand_ytrain, PCA_rand_ytest = train_test_split(PCA_rand_X, PCA_rand_y,
                                                                                    test_size = 0.2)

n_split = int(len(df_y) * 0.8)
PCA_Xtrain, PCA_ytrain = np.array(PCA_X)[:n_split, :], np.array(PCA_y)[:n_split] 
PCA_Xtest, PCA_ytest = np.array(PCA_X)[n_split:, :], np.array(PCA_y)[n_split:]



df_rand_Xtrain, df_rand_Xtest, df_rand_ytrain, df_rand_ytest = train_test_split(df_rand_X, df_rand_y,
                                                                                test_size = 0.2)

df_Xtrain, df_ytrain = np.array(df_X)[:n_split, :], np.array(df_y)[:n_split] 
df_Xtest, df_ytest = np.array(df_X)[n_split:, :], np.array(df_y)[n_split:]


#try benchmark models:
def rmse(prediction, yval): #this method calculates the metrics
    return np.sqrt(mean_squared_error(prediction, yval))

#1. Lasso Regression
def Lasso_GSCV(Xtrain, Xval, ytrain, yval):
    '''

    '''
    print('Lasso GridSearchCV: ', strftime('%d %b %Y %H:%M:%S', gmtime()))
    params = {'alpha': [1e-10, 1e-09, 1e-08, 1e-07, 1e-06, 1e-05, 1e-04, 1e-03, 1e-02, 1e-01],
              'max_iter': [1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7]}
    
    cv_sets = ShuffleSplit(n_splits = 5, test_size = 0.2, random_state = 0)
    t0 = time()

    regressor = Lasso()
    grid = GridSearchCV(estimator = regressor, param_grid = params,
                        scoring = 'neg_mean_squared_error', cv = cv_sets)
    
    grid = grid.fit(Xtrain, ytrain)
    
    test_score = rmse(grid.predict(Xval), yval)
    print('Time algo takes: {:.3f} seconds'.format(time() - t0))
    print('Train error: {:.4f} ({:.2f}%)'.format(np.sqrt(-grid.best_score_), np.sqrt(-grid.best_score_) / np.mean(ytrain) * 100))
    print('Test error: {:.4f} ({:.2f}%)'.format(test_score, test_score / np.mean(yval) * 100))
    
    print(grid.best_estimator_)
    print(grid.cv_results_)
    return grid.cv_results_

def Lasso_optimize(Xtrain, Xval, ytrain, yval):
    '''

    '''
    print('Lasso GridSearchCV: ', strftime('%d %b %Y %H:%M:%S', gmtime()))

    t0 = time()

    regressor = Lasso(alpha = 0.0001, max_iter = 1000)
    regressor.fit(Xtrain, ytrain)

    train_score = rmse(regressor.predict(Xtrain), ytrain)
    test_score = rmse(regressor.predict(Xval), yval)
    print('Time algo takes: {:.3f} seconds'.format(time() - t0))
    print('Train error: {:.4f} ({:.2f}%)'.format(train_score, train_score / np.mean(ytrain) * 100))
    print('Test error: {:.4f} ({:.2f}%)'.format(test_score, test_score / np.mean(yval) * 100))
    
    pass

def XGBR_GSCV(Xtrain, Xval, ytrain, yval):
    '''
    params = {'reg_alpha': [0.00001, 0.000025, 0.00005,0.000075,
                        0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007,
                        0.0008, 0.0009, 0.001, 0.0025, 0.005],
              'n_estimators': range(1500, 5000, 100),
              'learning_rate': [0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.055, 0.06, 0.065,
                                0.07, 0.075, 0.08, 0.085, 0.09, 0.1],
              'max_depth': range(1, 10)}
  
    '''
    print('XGBoost GridSearchCV: ', strftime('%d %b %Y %H:%M:%S', gmtime()))
    params = {'learning_rate': [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
              'max_depth': [2, 3, 4, 5, 6, 7, 8, 9, 10],
              'n_estimators': [600, 700, 800, 900, 1000, 1100, 1200, 1300],
              'reg_alpha': [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]}
    
    cv_sets = ShuffleSplit(n_splits = 5, test_size = 0.2, random_state = 0)
    clfier = xgb.XGBClassifier()
    acc_scorer = make_scorer(accuracy_score)
    t0 = time()
    grid = GridSearchCV(estimator = clfier, param_grid = params,
                        scoring = acc_scorer, cv = cv_sets)
    
    grid = grid.fit(Xtrain, ytrain)

    test_score = rmse(grid.predict(Xval), yval)
    print('Time algo takes: {:.3f} seconds'.format(time() - t0))
    print('Train accuracy: {:.4f}'.format(grid.best_score_))
    print('Test accuracy: {:.4f}'.format(test_score))
    
    print(grid.best_estimator_)
    print(grid.cv_results_)
    return grid.cv_results_

def XGBR_optimize(Xtrain, Xval, ytrain, yval):
    '''
    params = {'reg_alpha': [0.00001, 0.000025, 0.00005,0.000075,
                        0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007,
                        0.0008, 0.0009, 0.001, 0.0025, 0.005],
              'n_estimators': range(1500, 5000, 100),
              'learning_rate': [0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.055, 0.06, 0.065,
                                0.07, 0.075, 0.08, 0.085, 0.09, 0.1],
              'max_depth': range(1, 10)}
  
    '''
    print('XGBoost GridSearchCV: ', strftime('%d %b %Y %H:%M:%S', gmtime()))
    t0 = time()
    clfier = xgb.XGBClassifier(learning_rate = 0.001, max_depth = 5, n_estimators = 1000,
                               reg_alpha = 0.001)
    clfier.fit(Xtrain, ytrain)
    
    print('Time algo takes: {:.3f} seconds'.format(time() - t0))
    print('Train accuracy: {:.4f}'.format(accuracy_score(clfier.predict(Xtrain), ytrain)))
    print('Test accuracy: {:.4f}'.format(accuracy_score(clfier.predict(Xval), yval)))
    
    pass


def window_transform_series(X, y, window_size):
    # containers for input/output pairs
    X_result = []
    y_result = []
    #print(series)
    #print(window_size)
    for i in range(len(y) - window_size):
        X_result.append(X[i: i + window_size])
        y_result.append(y[i + window_size])
        #print(i)
        #print(series[i: i + window_size])
        #print(series[i + window_size])
    # reshape each

    X_result = np.asarray(X_result)
    X_result.shape = (np.shape(X_result)[0:3])

    y_result = np.asarray(y_result)
    y_result.shape = (len(y_result), 1)

    return X_result, y_result

#is it a good idea to tune LSTM using gridsearchCV??
#https://machinelearningmastery.com/tune-lstm-hyperparameters-keras-time-series-forecasting/
def LSTM_optimize(Xtrain, Xtest, ytrain, ytest, neurons, batch_size, epochs, repeat): #algorithms = , n_neighbors =
    #there are a couple of things we can vary:
    #1. epochs
    #2. neurons
    #3. batch_size
    #4. layers
    error_train = []
    error_test = []
    for i in range(repeat):
        model = Sequential()
        model.add(LSTM(neurons, input_shape = (Xtrain.shape[1], Xtrain.shape[2])))
        model.add(Dense(1))
        model.compile(loss = 'mean_squared_error', optimizer = 'adam')
        model.fit(Xtrain, ytrain, epochs = epochs, batch_size = batch_size, verbose=0, shuffle=False)
        
        rmse_train = np.sqrt(mean_squared_error(model.predict(Xtrain, batch_size = batch_size), ytrain))
        rmse_test = np.sqrt(mean_squared_error(model.predict(Xtest, batch_size = batch_size), ytest))
        error_train.append(rmse_train)
        error_test.append(rmse_test)
    return error_train, error_test

''' my target variables
0:'Adj Close 1day'
1:'Adj Close 5day'
2:'Adj Close 1day pct_change'
3:'Adj Close 5day pct_change'
4:'Adj Close 1day pct_change cls'
5:'Adj Close 5day pct_change cls'
'''

#benchmark models: get a rough idea on what type of error/accuracy we can achieve
if False: #Lasso Regression: run grid search cross validation to find the best parameters 
    variable = 'Adj Close 5day pct_change'
    results_dict = Lasso_GSCV(PCA_rand_Xtrain,
                              PCA_rand_Xtest,
                              PCA_rand_ytrain[variable],
                              PCA_rand_ytest[variable])


    result = pd.DataFrame()
    result['param_max_iter'] = results_dict['param_max_iter'].data
    result['param_alpha'] = results_dict['param_alpha'].data
    result['mean_train_score'] = results_dict['mean_train_score'].data
    result['mean_test_score'] = results_dict['mean_test_score'].data

    param_max_iter = result.groupby('param_max_iter').mean()
    param_max_iter['train avg'] = result.groupby('param_max_iter').mean()['mean_train_score']/df_rand_ytrain[variable].mean()
    param_max_iter['test avg'] = result.groupby('param_max_iter').mean()['mean_test_score']/df_rand_ytest[variable].mean()

    param_alpha = result.groupby('param_alpha').mean()
    param_alpha['train avg'] = result.groupby('param_alpha').mean()['mean_train_score']/df_rand_ytrain[variable].mean()
    param_alpha['test avg'] = result.groupby('param_alpha').mean()['mean_test_score']/df_rand_ytest[variable].mean()

    print('param_max_iter')
    print(param_max_iter)
    print('param_alpha')
    print(param_alpha)

if False: #Get the best possible results from Lasso Regression
    variable = 'Adj Close 5day pct_change'
    Lasso_optimize(PCA_rand_Xtrain,
                   PCA_rand_Xtest,
                   PCA_rand_ytrain[variable],
                   PCA_rand_ytest[variable])

    #no PCA
    #'Adj Close 1day': achieving Train error: 0.0206 (0.74%), Test error: 0.0195 (0.72%)
    #'Adj Close 5day': achieving Train error: 0.0444 (1.59%), Test error: 0.0477 (1.73%)
    #'Adj Close 1day pct_change': achieving Train error: 0.0246 (1640.96%), Test error: 0.0311 (6591.40%)
    #'Adj Close 5day pct_change': achieving Train error: 0.0572 (894.90%), Test error: 0.0528 (816.17%)

    #with PCA
    #'Adj Close 1day': achieving Train error: 0.0197 (0.71%), Test error: 0.0250 (0.88%)
    #'Adj Close 5day': achieving Train error: 0.0448 (1.60%), Test error: 0.0469 (1.75%)
    #'Adj Close 1day pct_change': achieving Train error: 0.0263 (2113.21%), Test error: 0.0247 (1672.30%)
    #'Adj Close 5day pct_change': achieving Train error: 0.0578 (840.08%), Test error: 0.0500 (1109.72%)


if False: #XGBoost Classification: run grid search cross validation to find the best parameters 
    variable = 'Adj Close 5day pct_change cls'
    results_dict = XGBR_GSCV(PCA_rand_Xtrain,
                             PCA_rand_Xtest,
                             PCA_rand_ytrain[variable],
                             PCA_rand_ytest[variable])

    result = pd.DataFrame()

    result['param_learning_rate'] = results_dict['param_learning_rate'].data
    result['param_max_depth'] = results_dict['param_max_depth'].data
    result['param_n_estimators'] = results_dict['param_n_estimators'].data
    result['param_reg_alpha'] = results_dict['param_reg_alpha'].data

    result['mean_train_score'] = results_dict['mean_train_score'].data
    result['mean_test_score'] = results_dict['mean_test_score'].data

    param_learning_rate = result.groupby('param_learning_rate').mean()
    param_max_depth = result.groupby('param_max_depth').mean()
    param_n_estimators = result.groupby('param_n_estimators').mean()
    param_reg_alpha = result.groupby('param_reg_alpha').mean()

    print('param_learning_rate')
    print(param_learning_rate)
    print('param_max_depth')
    print(param_max_depth)
    print('param_n_estimators')
    print(param_n_estimators)
    print('param_reg_alpha')
    print(param_reg_alpha)

if False: #Get the best possible results from XGBoost Classifier
    variable = 'Adj Close 1day pct_change cls'
    XGBR_optimize(df_rand_Xtrain,
                  df_rand_Xtest,
                  df_rand_ytrain[variable],
                  df_rand_ytest[variable])

    #no PCA
    #'Adj Close 1day pct_change cls': achieving Train accuracy: 0.5990, Test accuracy: 0.5011
    #'Adj Close 5day pct_change cls': achieving Train accuracy: 0.6457, Test accuracy: 0.5884

    #with PCA
    #'Adj Close 1day pct_change cls': achieving Train accuracy: 0.6476, Test accuracy: 0.5338
    #'Adj Close 5day pct_change cls': achieving Train accuracy: 0.6591, Test accuracy: 0.5611

#LSTM
if True:
    window_size = 5
    X_train, y_train = window_transform_series(df_Xtrain, df_ytrain[:, 2], window_size = window_size)
    X_test, y_test = window_transform_series(df_Xtest, df_ytest[:, 2], window_size = window_size)

    # X_train.shape: (3656, 5, 10) and y_train.shape: (3656, 1)
    #3656 is the number of rows, 5 is the batch_size/window_size, 10 is the number of features

    # NOTE: to use keras's RNN LSTM module our input must be reshaped to [samples, window size, stepsize]
    #test epochs first

    train_results = pd.DataFrame()
    test_results = pd.DataFrame()
    epochs = [10, 50, 100, 250, 500, 1000, 2000]
    for e in epochs:
        train_results[str(e)], test_results[str(e)] = LSTM_optimize(X_train, X_test, y_train, y_test,
                                                                    neurons = 5, batch_size = 5,
                                                                    epochs = e, repeat = 10)
        print('finished with ', e)

    print(train_results)

    
