import pandas as pd
import numpy as np
import pandas_datareader as pdr
import datetime
import quandl

from scipy.stats import norm, skew
from scipy import stats
from scipy.special import boxcox1p
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import matplotlib.pyplot as plt
import seaborn as sns

#we got a problem...
#yahoo and google API doesn't exist anymore. I'll need to pull updated info from quandl
#and compare those with historical data downloaded from 

def quandl_init(key):
    Quandlkey = ['7ieY8tq_kjzWx2-DiyGx', 'XrVxmKtfg2Fo3FG_NmtC', 'Jh3CAbmwaNP7YoqAN4FK']
    if key == None:
        quandl.ApiConfig.api_key = Quandlkey[0]
        return Quandlkey[0]
    else:
        if Quandlkey.index(key) == 2:
            quandl.ApiConfig.api_key = Quandlkey[0]
            return Quandlkey[0]
        else:
            quandl.ApiConfig.api_key = Quandlkey[Quandlkey.index(key) + 1]
            return Quandlkey[Quandlkey.index(key) + 1]

def extract_from_quandl(ticker, start_date=(2000, 1, 1), end_date = None):
    current_key = quandl_init(None)

    query_list = ['WIKI' + '/' + ticker + '.' + str(k) for k in range(1, 13)]
    #print(query_list)
    start_date = datetime.date(*start_date)
    #print(start_date)
    if end_date:
        end_date = datetime.date(*end_date)
    else:
        end_date = datetime.date.today()
 
    return quandl.get(query_list, returns = 'pandas',
                      start_date = start_date, end_date = end_date,
                      collapse = 'daily', order = 'asc')

def illustrate_datasets():
    tickers = ['AAPL']

    #get data from quandl, if necessary
    if False:
        for ticker in tickers:
            data = extract_from_quandl(ticker)
            data.to_csv('Data/' + ticker + '-quandl.csv')

        print('finished getting from Quandl')

    #comparing data sources (quandl vs. yahoo vs. NASDAQ)
    for ticker in tickers:
        #create a base dataframe
        df = pd.DataFrame(index = pd.date_range('2000/1/1', datetime.datetime.today(), freq='d'))
        #load yahoo data, already downloaded from finance.yahoo.com refresh if necessary
        data_yahoo = pd.read_csv('Data/' + ticker + '-yahoo.csv', index_col = 'Date')[['Adj Close', 'Volume']]
        data_yahoo.index = pd.to_datetime(data_yahoo.index)
        #extract quandl data, extracted through API
        data_quandl = pd.read_csv('Data/' + ticker + '-quandl.csv', index_col = 'Date')[['WIKI/' + ticker + ' - Adj. Close', 'WIKI/' + ticker + ' - Adj. Volume']]
        data_quandl.index = pd.to_datetime(data_quandl.index)
        #load NASDAQ data, already downloaded, refresh if necessary
        data_nasdaq = pd.read_csv('Data/' + ticker + '-nasdaq.csv', index_col = 'date', dtype = {'volume': float})[['close','volume']]
        data_nasdaq.index = pd.to_datetime(data_nasdaq.index)
        data_nasdaq.index = data_nasdaq.index.date #gotta do this cuz nasdaq is weird
        
        
        df = df.merge(data_yahoo[['Adj Close', 'Volume']], how = 'left', left_index = True, right_index = True)
        df = df.merge(data_quandl[['WIKI/' + ticker + ' - Adj. Close', 'WIKI/' + ticker + ' - Adj. Volume']], how = 'left', left_index = True, right_index = True)
        df = df.merge(data_nasdaq[['close', 'volume']], how = 'left', left_index = True, right_index = True)
        #using the yahoo as benchmark
        df['Adj Close Diff1'] = (df['Adj Close'] - df['WIKI/' + ticker + ' - Adj. Close']) / df['Adj Close']
        df['Adj Close Diff2'] = (df['Adj Close'] - df['close']) / df['Adj Close']
        df['Volume Diff1'] = (df['Volume'] - df['WIKI/' + ticker + ' - Adj. Volume']) / df['Volume']
        df['Volume Diff2'] = (df['Volume'] - df['volume']) / df['Volume']
        df.dropna(how = 'all', axis = 0, inplace = True)
        print(ticker)
        print('length: ', len(df))
        print('comparing yahoo and quandl, we are missing', sum(df['Volume Diff1'].isnull()), 'rows')
        print('comparing yahoo and nasdaq, we are missing', sum(df['Volume Diff2'].isnull()), 'rows')
        
        #initializing the figure
        fig = plt.figure(figsize = (16, 8))
        #we are going to have two charts/subplots. first chart will take up 4x4 space in a 5x4 grid
        ax1 = plt.subplot2grid((6,4),(0,0), rowspan = 4, colspan = 4)
        ax1.plot(df[['Adj Close']], label = 'yahoo', alpha = 0.5, color = 'r')
        ax1.plot(df[['WIKI/' + ticker + ' - Adj. Close']], label = 'quandl', alpha = 0.5, color = 'k')
        ax1.plot(df[['close']], label = 'nasdaq', alpha = 0.5, color = 'g')
        ax1.legend()
        plt.ylabel('Price')

        #for the second chart, it will take up 1x4 space in a 5x4 grid. it will also share the same axis as the first plot
        ax2 = plt.subplot2grid((6,4),(4,0), sharex = ax1, rowspan = 1, colspan = 4)
        ax2.bar(df.index, df['Volume'], label = 'yahoo', alpha = 0.5, color = 'r')
        ax2.bar(df.index, df['WIKI/' + ticker + ' - Adj. Volume'], label = 'quandl', alpha = 0.5, color = 'k')
        ax2.bar(df.index, df['volume'], label = 'nasdaq', alpha = 0.5, color = 'g')
        #ax2.axes.yaxis.set_ticklabels([]) #for some reason we are hiding the y axis lael for chart 2
        ax2.legend()
        plt.ylabel('Volume')
        
        ax3 = plt.subplot2grid((6,4),(5,0), sharex = ax1, rowspan = 1, colspan = 4)
        ax3.plot(df[['Adj Close Diff1']], label = 'close diff btw yho&quandl', alpha = 0.5, color = 'r')
        ax3.plot(df[['Adj Close Diff2']], label = 'close diff btw yho&nasdaq', alpha = 0.5, color = 'g')
        ax3.plot(df[['Volume Diff1']], label = 'vol diff btw yho&quandl', alpha = 0.5, color = 'k')
        ax3.plot(df[['Volume Diff2']], label = 'vol diff btw yho&nasdaq', alpha = 0.5, color = 'b')
        #ax3.axes.yaxis.set_ticklabels([]) #for some reason we are hiding the y axis lael for chart 2
        ax3.legend()

        #formating
        ax1.grid(True)
        ax2.grid(True)
        ax3.grid(True)

        plt.xlabel('Date')
        
        plt.tight_layout()
        plt.subplots_adjust(left = 0.09, bottom = 0.18, right = 0.97, top = 0.94, wspace = 0.20, hspace = 0)
        plt.suptitle(ticker + ' Stock Price')
        plt.setp(ax1.get_xticklabels(), visible = False) #removing the first chart's x label
            
        plt.show()
    pass


if __name__ == '__main__':
    ticker = 'AAPL'
    data_yahoo = pd.read_csv('Data/' + ticker + '-yahoo.csv', index_col = 'Date')
    data_yahoo.index = pd.to_datetime(data_yahoo.index)
    #here I'm trying to paint a shape of what's happening during the trading hours
    data_yahoo['Range'] = (data_yahoo['High'] - data_yahoo['Low'])/ data_yahoo['Open']
    data_yahoo['High'] = data_yahoo['High'] / data_yahoo['Open'] - 1
    data_yahoo['Low'] = data_yahoo['Low'] / data_yahoo['Open'] - 1
    data_yahoo['Open'] = data_yahoo['Open'] / data_yahoo['Close'].shift(1) - 1
    
    data_yahoo['MA5 Adj Close'] = data_yahoo['Adj Close'].rolling(window = 5).mean().shift(1)
    data_yahoo['MA5 Volume'] = data_yahoo['Volume'].rolling(window = 5).mean().shift(1)
    data_yahoo['MA5 Adj Close pct_change'] = data_yahoo['Adj Close'] / data_yahoo['MA5 Adj Close'] - 1
    data_yahoo['MA5 Volume pct_change'] = data_yahoo['Volume'] / data_yahoo['MA5 Volume'] - 1

    #this is what we are trying to predict
    #1. 1 day future price, boxcox1p transform
    #2. 5 days future price, boxcox1p transform
    #3. 1 day future price percentage change
    #4. 5 day future price percentage change
    #5. 1 day future price direction
    #6. 5 day future price direction
    data_yahoo['Adj Close 1day'] = data_yahoo['Adj Close'].shift(-1)
    data_yahoo['Adj Close 5day'] = data_yahoo['Adj Close'].shift(-5)
    data_yahoo['Adj Close 1day pct_change'] = data_yahoo['Adj Close 1day'] / data_yahoo['Adj Close'] - 1
    data_yahoo['Adj Close 5day pct_change'] = data_yahoo['Adj Close 5day'] / data_yahoo['Adj Close'] - 1
    data_yahoo['Adj Close 1day pct_change cls'] = data_yahoo['Adj Close 1day pct_change'].apply(lambda x: 1 if x >= 0 else 0)
    data_yahoo['Adj Close 5day pct_change cls'] = data_yahoo['Adj Close 5day pct_change'].apply(lambda x: 1 if x >= 0 else 0)
    data_yahoo.dropna(axis = 0, how = 'any', inplace = True)
    #print(data_yahoo.head(10))

    #let's look at the target variable distribution
    #plt.hist(data_yahoo['Adj Close 1day']) or plt.hist(data_yahoo['Adj Close 5day']) don't show much

    if False: #scaling isn't all that great for these two target variables
        for col_label in ['Adj Close 1day', 'Adj Close 5day']:
            #MM_Scaler = StandardScaler()
            #data = MM_Scaler.fit_transform(data_yahoo[col_label])
            #data = data - np.min(data)
            data = data_yahoo[col_label]
            lam = 0.0001
            
            fig, (ax1, ax2, ax3) = plt.subplots(ncols = 3, figsize = (15, 6))
            sns.distplot(data, fit = norm, ax = ax1)
            sns.distplot(boxcox1p(data, lam), fit = norm, ax = ax2)
            sns.distplot(np.log(data + lam), fit = norm, ax = ax3)
                
            (mu1, sigma1) = norm.fit(data)
            (mu2, sigma2) = norm.fit(boxcox1p(data, lam))
            (mu3, sigma3) = norm.fit(np.log(data + lam))
                
            ax1.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu1, sigma1),
                        'Skewness: {:.2f}'.format(skew(data))], loc = 'best')
            ax2.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu2, sigma2),
                        'Skewness: {:.2f}'.format(skew(boxcox1p(data, lam)))], loc = 'best')
            ax3.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu3, sigma3),
                        'Skewness: {:.2f}'.format(skew(np.log(data + lam)))], loc = 'best')
            ax1.set_ylabel('Frequency')
            ax1.set_title(col_label + ' Distribution')
            ax2.set_title(col_label + ' Box-Cox Transformed')
            ax3.set_title(col_label + ' Log Transformed')
            plt.show()

    if False: #scalers doesn't really work here either
        for col_label in ['Adj Close 1day pct_change', 'Adj Close 5day pct_change']:
            lam = 0.0001
            #MM_Scaler = MinMaxScaler()
            #data = MM_Scaler.fit_transform(data_yahoo[col_label])
            data = data_yahoo[col_label]
            data = data - np.min(data)
            
            fig, (ax1, ax2, ax3) = plt.subplots(ncols = 3, figsize = (15, 6))
            sns.distplot(data, fit = norm, ax = ax1)
            sns.distplot(boxcox1p(data, lam), fit = norm, ax = ax2)
            sns.distplot(np.log(data + lam), fit = norm, ax = ax3)
                
            (mu1, sigma1) = norm.fit(data)
            (mu2, sigma2) = norm.fit(boxcox1p(data, lam))
            (mu3, sigma3) = norm.fit(np.log(data + lam))
                
            ax1.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu1, sigma1),
                        'Skewness: {:.2f}'.format(skew(data))], loc = 'best')
            ax2.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu2, sigma2),
                        'Skewness: {:.2f}'.format(skew(boxcox1p(data, lam)))], loc = 'best')
            ax3.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu3, sigma3),
                        'Skewness: {:.2f}'.format(skew(np.log(data + lam)))], loc = 'best')
            ax1.set_ylabel('Frequency')
            ax1.set_title(col_label + ' Distribution')
            ax2.set_title(col_label + ' Box-Cox Transformed')
            ax3.set_title(col_label + ' Log Transformed')
            plt.show()

    if False: #transformations doesn't work
        for col_label in ['Adj Close 1day pct_change cls', 'Adj Close 5day pct_change cls']:
            data = data_yahoo[col_label]
            lam = 0.0001
            
            fig, (ax1, ax2, ax3) = plt.subplots(ncols = 3, figsize = (15, 6))
            sns.distplot(data, fit = norm, ax = ax1)
            sns.distplot(boxcox1p(data, lam), fit = norm, ax = ax2)
            sns.distplot(np.log(data + lam), fit = norm, ax = ax3)
                
            (mu1, sigma1) = norm.fit(data)
            (mu2, sigma2) = norm.fit(boxcox1p(data, lam))
            (mu3, sigma3) = norm.fit(np.log(data + lam))
                
            ax1.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu1, sigma1),
                        'Skewness: {:.2f}'.format(skew(data))], loc = 'best')
            ax2.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu2, sigma2),
                        'Skewness: {:.2f}'.format(skew(boxcox1p(data, lam)))], loc = 'best')
            ax3.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu3, sigma3),
                        'Skewness: {:.2f}'.format(skew(np.log(data + lam)))], loc = 'best')
            ax1.set_ylabel('Frequency')
            ax1.set_title(col_label + ' Distribution')
            ax2.set_title(col_label + ' Box-Cox Transformed')
            ax3.set_title(col_label + ' Log Transformed')
            plt.show()

    #let's look at the distribution between each independent variables
    if True:
        for col_label in ['Open', 'High', 'Low', 'Range', 'Adj Close', 'Volume', 'MA5 Adj Close',
                          'MA5 Volume', 'MA5 Adj Close pct_change', 'MA5 Volume pct_change']:
            data = data_yahoo[col_label]
            if np.min(data) < 0:
                data = data - np.min(data)
            
            lam = 0.0001
            
            fig, (ax1, ax2, ax3) = plt.subplots(ncols = 3, figsize = (15, 6))
            sns.distplot(data, fit = norm, ax = ax1)
            sns.distplot(boxcox1p(data, lam), fit = norm, ax = ax2)
            sns.distplot(np.log(data + lam), fit = norm, ax = ax3)
                
            (mu1, sigma1) = norm.fit(data)
            (mu2, sigma2) = norm.fit(boxcox1p(data, lam))
            (mu3, sigma3) = norm.fit(np.log(data + lam))
                
            ax1.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu1, sigma1),
                        'Skewness: {:.2f}'.format(skew(data))], loc = 'best')
            ax2.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu2, sigma2),
                        'Skewness: {:.2f}'.format(skew(boxcox1p(data, lam)))], loc = 'best')
            ax3.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu3, sigma3),
                        'Skewness: {:.2f}'.format(skew(np.log(data + lam)))], loc = 'best')
            ax1.set_ylabel('Frequency')
            ax1.set_title(col_label + ' Distribution')
            ax2.set_title(col_label + ' Box-Cox Transformed')
            ax3.set_title(col_label + ' Log Transformed')
            plt.show()
        
