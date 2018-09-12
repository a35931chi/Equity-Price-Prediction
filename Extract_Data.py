import pandas as pd
import numpy as np
import pandas_datareader as pdr
import datetime
import quandl

from scipy.stats import norm, skew
from scipy import stats
from scipy.special import boxcox1p
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import seaborn as sns
import visuals as vs

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


def extract_data(ticker, pca = None):

    #all these don't have anything to do with whether a PCA exist or not
    data_yahoo = pd.read_csv('Data/' + ticker + '-yahoo.csv', index_col = 'Date')
    data_yahoo.index = pd.to_datetime(data_yahoo.index)
    
    #here I'm trying to paint a shape of what's happening during the trading hours, these are independent features
    #range describes the min/max distance per open price
    data_yahoo['Range'] = (data_yahoo['High'] - data_yahoo['Low'])/ data_yahoo['Open']
    #high is a percentage of open
    data_yahoo['High'] = data_yahoo['High'] / data_yahoo['Open'] - 1
    #low is a percentage of open
    data_yahoo['Low'] = data_yahoo['Low'] / data_yahoo['Open'] - 1
    #open is a percentage of previous day's close
    data_yahoo['Open'] = data_yahoo['Open'] / data_yahoo['Close'].shift(1) - 1
    #previous 5 days moving average (adj close)
    data_yahoo['MA5 Adj Close'] = data_yahoo['Adj Close'].rolling(window = 5).mean().shift(1)
    #previous 5 days moving average (volume)
    data_yahoo['MA5 Volume'] = data_yahoo['Volume'].rolling(window = 5).mean().shift(1)
    #% change vs. previous 5 days (adj close)
    data_yahoo['MA5 Adj Close pct_change'] = data_yahoo['Adj Close'] / data_yahoo['MA5 Adj Close'] - 1
    #% change vs. previous 5 days (volume)
    data_yahoo['MA5 Volume pct_change'] = data_yahoo['Volume'] / data_yahoo['MA5 Volume'] - 1

    print(data_yahoo.tail(20))
    what = input('what')
    
    #this is what we are trying to predict (targets)
    #1. 1 day future price
    data_yahoo['Adj Close 1day'] = data_yahoo['Adj Close'].shift(-1)
    #2. 5 days future price
    data_yahoo['Adj Close 5day'] = data_yahoo['Adj Close'].shift(-5)
    #data_yahoo['Adj Close 10day'] = data_yahoo['Adj Close'].shift(-10)
    #3. 1 day future price percentage change
    data_yahoo['Adj Close 1day pct_change'] = data_yahoo['Adj Close 1day'] / data_yahoo['Adj Close'] - 1
    #4. 5 day future price percentage change
    data_yahoo['Adj Close 5day pct_change'] = data_yahoo['Adj Close 5day'] / data_yahoo['Adj Close'] - 1
    #data_yahoo['Adj Close 10day pct_change'] = data_yahoo['Adj Close 10day'] / data_yahoo['Adj Close'] - 1
    #5. 1 day future price direction
    data_yahoo['Adj Close 1day pct_change cls'] = data_yahoo['Adj Close 1day pct_change'].apply(lambda x: 1 if x >= 0 else 0)
    #6. 5 day future price direction
    data_yahoo['Adj Close 5day pct_change cls'] = data_yahoo['Adj Close 5day pct_change'].apply(lambda x: 1 if x >= 0 else 0)
    #data_yahoo['Adj Close 10day pct_change cls'] = data_yahoo['Adj Close 10day pct_change'].apply(lambda x: 1 if x >= 0 else 0)

    data_yahoo.dropna(axis = 0, how = 'any', inplace = True)
    print(data_yahoo.head(20))
    what = input('what')

    #let's look at the target variable distribution
    if False: #scaling isn't all that great for these two target variables
        for col_label in ['Adj Close 1day', 'Adj Close 5day']:
            lam = 0.0001
            #scaler = StandardScaler()
            #data = scaler.fit_transform(data_yahoo[col_label])
            data = data_yahoo[col_label]
            if np.min(data) < 0:
                data = data - np.min(data)
            ''' 
            no scaler:
            1.2165656107790856 -0.06554419693948103 -0.2485500333952623
            1.2147780477183334 -0.06797363864363892 -0.25105816533149256
            
            MinMax:
            1.2165656107790865 0.9905547643484544 -0.6092542377635981
            1.2147780477183334 0.9885749885051007 -0.6115631693413965
            
            Standard:
            1.216565610779086 0.7273346450678947 -0.6463496872857882
            1.214778047718333 0.7258861448434313 -0.6485618913384967
            
            Adj Close 1day - no scaler boxcox
            Adj Close 5day - no scaler boxcox
            '''
            
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
            print(skew(data), skew(boxcox1p(data, lam)), skew(np.log(data + lam)))
            ax1.set_title(col_label + ' Distribution')
            ax2.set_title(col_label + ' Box-Cox Transformed')
            ax3.set_title(col_label + ' Log Transformed')
            plt.show()

    if False: #scalers doesn't really work here either
        for col_label in ['Adj Close 1day pct_change', 'Adj Close 5day pct_change']:
            lam = 0.0001
            #scaler = StandardScaler()
            #data = scaler.fit_transform(data_yahoo[col_label])
            data = data_yahoo[col_label]
            if np.min(data) < 0:
                data = data - np.min(data)
            ''' 
            no scaler:
            -1.6510040307386993 -3.041993709001984 -55.25486882951101
            -0.9408177644672319 -1.8326191132390537 -29.740251304355382
            
            MinMax:
            -1.6510040307386906 -3.7210597124936196 -56.219015977319174
            -0.9408177644672386 -1.928686775469499 -30.170597099885942
            
            Standard:
            -1.6510040307386935 -23.430168647942985 -61.022779357622056
            -0.9408177644672379 -7.476432811167501 -39.192139404540846
            
            Adj Close 1day pct_change - no scaler no transform
            Adj Close 5day pct_change - no scaler no transform
            '''
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
            print(skew(data), skew(boxcox1p(data, lam)), skew(np.log(data + lam)))
            ax1.set_title(col_label + ' Distribution')
            ax2.set_title(col_label + ' Box-Cox Transformed')
            ax3.set_title(col_label + ' Log Transformed')
            plt.show()

    if False: #transformations doesn't work
        for col_label in ['Adj Close 1day pct_change cls', 'Adj Close 5day pct_change cls']:

            lam = 0.0001
            #scaler = StandardScaler()
            #data = scaler.fit_transform(data_yahoo[col_label])
            data = data_yahoo[col_label]
            if np.min(data) < 0:
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
            print(skew(data), skew(boxcox1p(data, lam)), skew(np.log(data + lam)))
            ax1.set_title(col_label + ' Distribution')
            ax2.set_title(col_label + ' Box-Cox Transformed')
            ax3.set_title(col_label + ' Log Transformed')
            plt.show()

    #let's look at the distribution between each independent variables
    if False:
        for col_label in ['Open', 'High', 'Low', 'Range', 'Adj Close', 'Volume', 'MA5 Adj Close',
                          'MA5 Volume', 'MA5 Adj Close pct_change', 'MA5 Volume pct_change']:
            lam = 0.0001
            MM_Scaler = StandardScaler()
            data = MM_Scaler.fit_transform(data_yahoo[col_label])
            #data = data_yahoo[col_label]
            if np.min(data) < 0:
                data = data - np.min(data)
            '''
            no scaler:
            -5.81225631547921 -9.146269872594456 -62.02985068281074
            2.712481343056322 2.5700673022819003 -0.9250758631542217
            -2.5660281111479226 -2.748341308395658 -33.43627464332718
            2.0711648325652057 1.9393218387203244 0.11413402414840106
            1.2168876669938415 -0.06494740962703247 -0.24793607926031352
            2.964204574183921 -0.053455194367689286 -0.05363840820122992
            1.2162667302493857 -0.06319870260334028 -0.24591172238801545
            1.612323763816607 -0.13175043131958308 -0.13190749719487718
            -1.5597704161777437 -2.5595298869477796 -37.90656860268758
            6.2660356551310485 1.3701314015201278 -1.5385880097432818
            
            MinMax:
            -5.8122563154792015 -11.082155965519698 -62.60899490983156
            2.7124813430563224 2.1099552360244025 -1.5007847275278556
            -2.566028111147925 -3.2923115321333882 -39.97787685974701
            2.071164832565206 1.63981390276237 -0.5591602425043781
            1.216887666993841 0.9909781433323644 -0.6086980942779617
            2.96420457418392 2.1627318198895265 -0.44700320966425255
            1.2162667302493853 0.9896212018472764 -0.6191447177436266
            1.6123237638166064 1.2209628689973062 -0.8754504958756806
            -1.5597704161777388 -2.9341885148669853 -39.23072118617314
            6.26603565513105 4.0674910496926024 -0.6555394360531169

            Standard:
            -5.812256315479212 -43.76335777947685 -65.00115946436995
            2.7124813430563215 0.9592105401947402 -2.3878118031831503
            -2.566028111147922 -7.668800386424772 -47.99647262001495
            2.0711648325652066 0.6949862598819779 -0.7304334255376943
            1.2168876669938415 0.7276509004530627 -0.6458190217616919
            2.9642045741839196 0.7457660928003991 -0.6036534963638037
            1.2162667302493853 0.7279810838194929 -0.6604003387555384
            1.6123237638166061 0.47908958845988775 -0.9756621700306726
            -1.5597704161777395 -11.379256076858951 -47.849396206857875
            6.2660356551310485 0.9336930967122643 -1.872136860642601

            Open - no scaler no transform
            High - StandardScaler boxcox transform
            Low - no scaler no transform
            Range - no scaler log transform
            Adj Close - no scaler boxcox transform
            Volume - no scaler boxcox transform
            MA5 Adj Close - no scaler boxcox transform
            MA5 Volume - no scaler boxcox transform
            MA5 Adj Close pct_change - no scaler no transform
            MA5 Volume pct_change - MinMaxScaler log transform
            '''            
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
            print(skew(data), skew(boxcox1p(data, lam)), skew(np.log(data + lam)))
            ax1.set_ylabel('Frequency')
            ax1.set_title(col_label + ' Distribution')
            ax2.set_title(col_label + ' Box-Cox Transformed')
            ax3.set_title(col_label + ' Log Transformed')
            plt.show()
        
    #so what do we need to transform?
    #no scaler no transform
    #these transofrmations don't need PCA either
    lam = 0.0001
    col_names = ['Adj Close 1day pct_change', 'Adj Close 5day pct_change', 'Adj Close 1day pct_change cls',
                 'Adj Close 5day pct_change cls', 'Open', 'Low', 'MA5 Adj Close pct_change']
    #no scaler, boxcox transform
    col_names = ['Adj Close 1day', 'Adj Close 5day', 'Adj Close',
                 'MA5 Adj Close', 'MA5 Volume', 'Volume']
    for col_name in col_names:
        data_yahoo[col_name] = boxcox1p(data_yahoo[col_name], lam)

    #no scaler, log transform
    data_yahoo['Range'] = np.log(data_yahoo['Range'] + lam)
    #StandardScaler, boxcox transform
    SS_scaler = StandardScaler()
    data_yahoo['High'] = boxcox1p(SS_scaler.fit_transform(data_yahoo['High']), lam)
    #MinMaxScaler, log transform    
    MM_scaler = MinMaxScaler()
    data_yahoo['MA5 Volume pct_change'] = np.log(MM_scaler.fit_transform(data_yahoo['MA5 Volume pct_change']) + lam)


    #let's look at heatmaps
    if False: #correlation X vs. ylog
        print(data_yahoo.head(20))
        corrmat = data_yahoo.corr()
        plt.subplots(figsize = (12, 9))
        g = sns.heatmap(corrmat, vmax = 0.9, square = True, annot = True, annot_kws={'size': 8})
        g.set_yticklabels(g.get_yticklabels(), rotation = 0, fontsize = 8)
        g.set_xticklabels(g.get_xticklabels(), rotation = 90, fontsize = 8)
        plt.title('Correlation Matrix/Heatmap Numerical Features vs. Targets')
        plt.tight_layout()
        plt.show()

    #let's also try PCA
    train = data_yahoo[['Open', 'High', 'Low', 'Range', 'Adj Close', 'Volume', 'MA5 Adj Close',
                        'MA5 Volume', 'MA5 Adj Close pct_change', 'MA5 Volume pct_change']]

    if pca == None:
        pca = PCA(n_components = 7)
        trainPCA = pd.DataFrame(pca.fit_transform(train))

    else:
        trainPCA = pd.DataFrame(pca.transform(train))

    PCA_data_yahoo = pd.DataFrame.copy(trainPCA)
    PCA_data_yahoo.columns = ['Dimension ' + str(i) for i in range(1,8)]
    for target in ['Adj Close 1day', 'Adj Close 5day', 'Adj Close 1day pct_change', 'Adj Close 5day pct_change',
                    'Adj Close 1day pct_change cls', 'Adj Close 5day pct_change cls']:
        PCA_data_yahoo[target] = data_yahoo.reset_index()[target]
        
        
    
    if False: #show PCA results, cumulative power, and heatmap
        pca_results = vs.pca_results(train, pca)
        plt.show() 
        ys = pca.explained_variance_ratio_
        xs = np.arange(1, len(ys)+1)
        plt.plot(xs, np.cumsum(ys), '-o')
        for label, x, y in zip(np.cumsum(ys), xs, np.cumsum(ys)):
            plt.annotate('{:.2f}%'.format(label * 100),
                xy = (x, y), xytext=(30, -20),
                textcoords = 'offset points', ha = 'right', va = 'bottom',
                bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
                arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))
    
        plt.ylabel('Cumulative Explained Variance')
        plt.xlabel('Dimensions')
        plt.title('PCA - Total Explained Variance by # fo Dimensions')
        plt.tight_layout()
        plt.show()
    
        g = sns.heatmap(temp.corr(), annot = True, annot_kws={'size': 8})
        g.set_yticklabels(g.get_yticklabels(), rotation = 0, fontsize = 8)
        g.set_xticklabels(g.get_xticklabels(), rotation = 90, fontsize = 8)
        plt.title('PCA Correlation Matrix/Heatmap')
        plt.tight_layout()
        #plt.savefig('Charts/PCA heatmap.png')
        plt.show()

    #export pca, pca dataset, and original dataset
    return pca, PCA_data_yahoo, data_yahoo
        

if __name__ == '__main__':
    #illustrate_datasets()
    pca, PCA_df, df = extract_data('BTC-USD', None)
