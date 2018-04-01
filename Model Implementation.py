import numpy as np
import pandas as pd
from time import gmtime, strftime, time

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, ShuffleSplit, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler

from sklearn.metrics import mean_squared_error

from sklearn.linear_model import Lasso
import xgboost as xgb

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

PCA_X = PCA_df[['Dimension 1', 'Dimension 2', 'Dimension 3', 'Dimension 4',
                   'Dimension 5', 'Dimension 6', 'Dimension 7']]
PCA_y = PCA_df[['Adj Close 1day', 'Adj Close 5day',
                'Adj Close 1day pct_change', 'Adj Close 5day pct_change',
                'Adj Close 1day pct_change cls', 'Adj Close 5day pct_change cls']]

df = shuffle(df, random_state = 0)
df_X = df[['Open', 'High', 'Low', 'Adj Close', 'Volume', 'Range', 'MA5 Adj Close', 'MA5 Volume',
           'MA5 Adj Close pct_change', 'MA5 Volume pct_change']]
df_y = df[['Adj Close 1day', 'Adj Close 5day', 'Adj Close 1day pct_change',
           'Adj Close 5day pct_change', 'Adj Close 1day pct_change cls',
           'Adj Close 5day pct_change cls']]

#split into train, test, validation sets
PCA_Xtrain, PCA_Xtest, PCA_ytrain, PCA_ytest = train_test_split(PCA_X, PCA_y, test_size = 0.2,
                                                                random_state = 42)

PCA_Xtrain, PCA_Xval, PCA_ytrain, PCA_yval = train_test_split(PCA_Xtrain, PCA_ytrain, test_size = 0.2,
                                                                random_state = 42)


df_Xtrain, df_Xtest, df_ytrain, df_ytest = train_test_split(df_X, df_y, test_size = 0.2,
                                                            random_state = 42)
df_Xtrain, df_Xval, df_ytrain, df_yval = train_test_split(df_Xtrain, df_ytrain, test_size = 0.2,
                                                          random_state = 42)


#try benchmark models:
def rmse(prediction, yval): #this method calculates the metrics
    return np.sqrt(mean_squared_error(prediction, yval))

#1. Lasso Regression
def Lasso_GSCV(Xtrain, Xval, ytrain, yval):
    '''
    no PCA:
    predicting: Adj Close 1day
    Time algo takes: 26.254 seconds
    Train error: 0.0216 (0.78%)
    Test error: 0.0178 (0.65%)
    Lasso(alpha=0.0001, copy_X=True, fit_intercept=True, max_iter=100.0,
       normalize=False, positive=False, precompute=False, random_state=None,
       selection='cyclic', tol=0.0001, warm_start=False)

    predicting: Adj Close 5day
    Time algo takes: 51.389 seconds
    Train error: 0.0460 (1.66%)
    Test error: 0.0417 (1.53%)
    Lasso(alpha=1e-05, copy_X=True, fit_intercept=True, max_iter=100.0,
       normalize=False, positive=False, precompute=False, random_state=None,
       selection='cyclic', tol=0.0001, warm_start=False)

    predicting: Adj Close 1day pct_change
    Time algo takes: 57.937 seconds
    Train error: 0.0277 (5294.01%)
    Test error: 0.0234 (1285.35%)
    Lasso(alpha=0.001, copy_X=True, fit_intercept=True, max_iter=10.0,
       normalize=False, positive=False, precompute=False, random_state=None,
       selection='cyclic', tol=0.0001, warm_start=False)

    predicting: Adj Close 5day pct_change
    Time algo takes: 55.721 seconds
    Train error: 0.0568 (983.14%)
    Test error: 0.0545 (869.17%)
    Lasso(alpha=1e-05, copy_X=True, fit_intercept=True, max_iter=10.0,
       normalize=False, positive=False, precompute=False, random_state=None,
       selection='cyclic', tol=0.0001, warm_start=False)

    PCA:
    predicting: Adj Close 1day
    Lasso GridSearchCV:  01 Apr 2018 00:10:49
    Time algo takes: 0.775 seconds
    Train error: 0.0218 (0.79%)
    Test error: 0.0181 (0.66%)
    Lasso(alpha=1e-05, copy_X=True, fit_intercept=True, max_iter=10.0,
       normalize=False, positive=False, precompute=False, random_state=None,
       selection='cyclic', tol=0.0001, warm_start=False)

    predicting: Adj Close 5day
    Lasso GridSearchCV:  01 Apr 2018 00:10:10
    Time algo takes: 0.700 seconds
    Train error: 0.0461 (1.66%)
    Test error: 0.0419 (1.53%)
    Lasso(alpha=1e-08, copy_X=True, fit_intercept=True, max_iter=10.0,
       normalize=False, positive=False, precompute=False, random_state=None,
       selection='cyclic', tol=0.0001, warm_start=False)

    predicting: Adj Close 1day pct_change
    Lasso GridSearchCV:  01 Apr 2018 00:13:19
    Time algo takes: 0.935 seconds
    Train error: 0.0277 (5294.77%)
    Test error: 0.0234 (1285.35%)
    Lasso(alpha=0.001, copy_X=True, fit_intercept=True, max_iter=10.0,
       normalize=False, positive=False, precompute=False, random_state=None,
       selection='cyclic', tol=0.0001, warm_start=False)

    predicting: Adj Close 5day pct_change
    Lasso GridSearchCV:  01 Apr 2018 00:39:01
    Time algo takes: 0.911 seconds
    Train error: 0.0568 (982.59%)
    Test error: 0.0545 (869.70%)
    Lasso(alpha=1e-10, copy_X=True, fit_intercept=True, max_iter=10.0,
       normalize=False, positive=False, precompute=False, random_state=None,
       selection='cyclic', tol=0.0001, warm_start=False)

    '''
    print('Lasso GridSearchCV: ', strftime('%d %b %Y %H:%M:%S', gmtime()))
    params = {'alpha': [1e-10, 1e-09, 1e-08, 1e-07, 1e-06, 1e-05, 1e-04, 1e-03],
              'max_iter': [1e1, 1e2, 1e3, 1e4, 1e5]}
    
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
    #print(grid.cv_results_)
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
    no PCA:
    Adj Close 1day pct_change cls:
    Time algo takes: 1720.520 seconds
    Train error: 0.6880 (133.59%)
    Test error: 0.7008 (130.05%)
    XGBClassifier(base_score=0.5, colsample_bylevel=1, colsample_bytree=1,
           gamma=0, learning_rate=0.1, max_delta_step=0, max_depth=6,
           min_child_weight=1, missing=None, n_estimators=1000, nthread=-1,
           objective='binary:logistic', reg_alpha=1e-05, reg_lambda=1,
           scale_pos_weight=1, seed=0, silent=True, subsample=1)

    Adj Close 5day pct_change cls:
    
    PCA:
    Adj Close 1day pct_change cls:
    

    Adj Close 5day pct_change cls:
    
    '''
    print('XGBoost GridSearchCV: ', strftime('%d %b %Y %H:%M:%S', gmtime()))
    params = {'learning_rate': [0.01, 0.05, 0.1, 0.5, 1],
              'max_depth': [4, 5, 6, 7, 8],
              'n_estimators': [500, 600, 700, 800, 900, 1000, 1100],
              'reg_alpha': [1e-7, 1e-6, 1e-5, 5e-5]}
    
    cv_sets = ShuffleSplit(n_splits = 5, test_size = 0.2, random_state = 0)
    regressor = xgb.XGBClassifier()
    t0 = time()
    grid = GridSearchCV(estimator = regressor, param_grid = params,
                        scoring = 'neg_mean_squared_error', cv = cv_sets)
    
    grid = grid.fit(Xtrain, ytrain)

    test_score = rmse(grid.predict(Xval), yval)
    print('Time algo takes: {:.3f} seconds'.format(time() - t0))
    print('Train error: {:.4f} ({:.2f}%)'.format(np.sqrt(-grid.best_score_), np.sqrt(-grid.best_score_) / np.mean(ytrain) * 100))
    print('Test error: {:.4f} ({:.2f}%)'.format(test_score, test_score / np.mean(yval) * 100))
    
    print(grid.best_estimator_)
    #print(grid.cv_results_)
    pass

def Lasso_Robust(Xtrain, Xval, ytrain, yval):
    '''
    Best Model:
    Lasso(alpha=0.0006, copy_X=True, fit_intercept=True, max_iter=100,
       normalize=False, positive=False, precompute=False, random_state=None,
       selection='cyclic', tol=0.0001, warm_start=False)
    '''
    num_cols = ['Dimension 1', 'Dimension 2', 'Dimension 3', 'Dimension 4',
                'Dimension 5', 'Dimension 6', 'Dimension 7']
    Xtrain.reset_index(inplace = True)
    Xval.reset_index(inplace = True)
    Xtrain.drop(['index'], axis = 1, inplace = True)
    Xval.drop(['index'], axis = 1, inplace = True)
    
    ytrain.reset_index(drop = True, inplace = True)
    yval.reset_index(drop = True, inplace = True)
    
    n_folds = 10
    
    if True: #try different folds
        regressor = make_pipeline(RobustScaler(), Lasso(alpha = 0.0006, max_iter = 100))
        kf = KFold(n_folds, shuffle = True)
        rmse = np.sqrt(-cross_val_score(regressor, Xtrain.values, ytrain, scoring = 'neg_mean_squared_error', cv = kf))
        plt.plot(rmse)
        plt.xlabel('Kth Fold')
        plt.ylabel('RMSE')
        plt.title('Kth Fold vs. RMSE')
        plt.axhline(np.mean(rmse), linestyle = ':', color = 'r', label = 'mean RMSE')
        plt.legend()
        plt.tight_layout()
        plt.savefig('RMSE for each KFold.png')
        plt.show()
        
    if True: #try different random states
        mean = []
        std = []
        for i in range(20):
            regressor = make_pipeline(RobustScaler(), Lasso(alpha = 0.0006, max_iter = 100))
            kf = KFold(n_folds, shuffle = True)
            rmse = np.sqrt(-cross_val_score(regressor, Xtrain.values, ytrain, scoring = 'neg_mean_squared_error', cv = kf))
            mean.append(np.mean(rmse))
            std.append(np.std(rmse))
        fig, (ax1, ax2) = plt.subplots(ncols = 2, figsize = (10, 5))
        ax1.plot(mean)
        ax1.axhline(np.mean(mean), linestyle = ':', color = 'r')
        ax1.set_xlabel('#th iteration')
        ax1.set_ylabel('mean RMSE')
        ax1.set_title('20 K-Fold results: mean RMSE')
        ax2.plot(std)
        ax2.axhline(np.mean(std), linestyle = ':', color = 'r')
        ax2.set_xlabel('#th iteration')
        ax2.set_ylabel('std RMSE')
        ax2.set_title('20 K-Fold results: std RMSE')
        plt.tight_layout()
        plt.savefig('avg RMSE for diff random state.png')
        plt.show()

    #try small changes to the dataset
    if True: 
        #deletion observations points
        yidx = Xtrain.shape[0]
        error = []
        fracs = [0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.15, 0.2]

        for frac in fracs: 
            Xtrain_copy = pd.DataFrame.copy(Xtrain)
            ytrain_copy = pd.DataFrame.copy(ytrain)
            cidx = np.random.choice(yidx, round(0.05 * yidx))
            Xtrain_copy.drop(Xtrain_copy.index[cidx], inplace = True)
            ytrain_copy.drop(ytrain_copy.index[cidx], inplace = True)

            scaler = RobustScaler()
            Xtrainscaled = scaler.fit_transform(Xtrain_copy)
            Xvalscaled = scaler.transform(Xval)

            regressor = Lasso(alpha = 0.0006, copy_X = True, fit_intercept = True,
                              max_iter = 100, normalize = False, positive = False,
                              precompute = False, random_state = None, selection = 'cyclic',
                              tol = 0.0001, warm_start = False)
    
            regressor.fit(Xtrainscaled, ytrain_copy)
            test_score = np.sqrt(mean_squared_error(regressor.predict(Xvalscaled), yval))
            error.append(test_score)

            print('{:.0f}% data deleted, Test error: {:.4f} ({:.2f}%)'.format(frac * 100, test_score, test_score / np.mean(yval) * 100))
        plt.plot(fracs, error)
        plt.annotate('{:.4f}, {:.2f}% of target'.format(test_score, test_score / np.mean(yval) * 100),
                xy=(frac, test_score), xytext=(30, -20),
                textcoords='offset points', ha='right', va='bottom',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))
        #plt.ylim(ymin = 0)
        plt.axhline(0.1017, linestyle = ':', color = 'r', label = 'benchmark')
        plt.title('Test RMSE vs. % data removed')
        plt.xlabel('% data removed')
        plt.ylabel('Test RMSE')
        plt.legend()
        plt.tight_layout()
        plt.savefig('test RMSE vs % data removed.png')
        plt.show()
        
    if True:
        #select some observations and mutiply some points by 10
        yidx = Xtrain.shape[0]
        error_m = []
        error_d = []
        fracs = [0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.15, 0.2]

        for frac in fracs: 
            Xtrain_copy = pd.DataFrame.copy(Xtrain)

            cidx = np.random.choice(yidx, round(frac * yidx))
            Xtrain_copy.ix[cidx , num_cols] = Xtrain_copy.ix[cidx , num_cols] * 10
            
            scaler = RobustScaler()
            Xtrainscaled = scaler.fit_transform(Xtrain_copy)
            Xvalscaled = scaler.transform(Xval)

            regressor = Lasso(alpha = 0.0006, copy_X = True, fit_intercept = True,
                              max_iter = 100, normalize = False, positive = False,
                              precompute = False, random_state = None, selection = 'cyclic',
                              tol = 0.0001, warm_start = False)
    
            regressor.fit(Xtrainscaled, ytrain)
            test_scorem = np.sqrt(mean_squared_error(regressor.predict(Xvalscaled), yval))
            error_m.append(test_scorem)
            print('{:.0f}% data scaled up, Test error: {:.4f} ({:.2f}%)'.format(frac * 100, test_scorem, test_scorem / np.mean(yval) * 100))

        for frac in fracs: 
            Xtrain_copy = pd.DataFrame.copy(Xtrain)

            cidx = np.random.choice(yidx, round(frac * yidx))
            Xtrain_copy.ix[cidx , num_cols] = Xtrain_copy.ix[cidx , num_cols] / 10
            
            scaler = RobustScaler()
            Xtrainscaled = scaler.fit_transform(Xtrain_copy)
            Xvalscaled = scaler.transform(Xval)

            regressor = Lasso(alpha = 0.0006, copy_X = True, fit_intercept = True,
                              max_iter = 100, normalize = False, positive = False,
                              precompute = False, random_state = None, selection = 'cyclic',
                              tol = 0.0001, warm_start = False)
    
            regressor.fit(Xtrainscaled, ytrain)
            test_scored = np.sqrt(mean_squared_error(regressor.predict(Xvalscaled), yval))
            error_d.append(test_scored)
            print('{:.0f}% data scaled up, Test error: {:.4f} ({:.2f}%)'.format(frac * 100, test_scored, test_scored / np.mean(yval) * 100))
            
        plt.plot(fracs, error_m, label = 'some data scaled up')
        plt.plot(fracs, error_d, label = 'some data scaled down')
        plt.annotate('{:.4f}, {:.2f}% of target'.format(test_scorem, test_scorem / np.mean(yval) * 100),
                xy=(frac, test_scorem), xytext=(30, -20),
                textcoords='offset points', ha='right', va='bottom',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))
        plt.annotate('{:.4f}, {:.2f}% of target'.format(test_scored, test_scored / np.mean(yval) * 100),
                xy=(frac, test_scored), xytext=(30, -20),
                textcoords='offset points', ha='right', va='bottom',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))
        
        #plt.ylim(ymin = 0)
        plt.axhline(0.1017, linestyle = ':', color = 'r', label = 'benchmark')
        plt.title('Test RMSE vs. % data altered')
        plt.xlabel('% data altered')
        plt.ylabel('Test RMSE')
        plt.legend()
        plt.tight_layout()
        plt.savefig('test RMSE vs % data scaled.png')
        plt.show()

    pass

#Lasso_GSCV(df_Xtrain, df_Xval, df_ytrain['Adj Close 5day pct_change'],
#           df_yval['Adj Close 5day pct_change'])

XGBR_GSCV(df_Xtrain, df_Xval, df_ytrain['Adj Close 1day pct_change cls'],
          df_yval['Adj Close 1day pct_change cls'])
