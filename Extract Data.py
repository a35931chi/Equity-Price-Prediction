import pandas as pd
import pandas_datareader as pdr
import datetime
import quandl
import matplotlib.pyplot as plt

#we got a problem...
#yahoo and google API doesn't exist anymore. I'll need to pull updated info from quandl
#and compare those with historical data downloaded from 

quandl.ApiConfig.api_key = 'Jh3CAbmwaNP7YoqAN4FK'
quandl.get_table('WIKI/PRICES', date='1999-11-18', ticker='A')

def extract_from_yahoo(ticker):
    data = pdr.get_data_yahoo(symbols = ticker,
                              start = datetime(2000, 1, 1),
                              end = datetime(2012, 1, 1))
    return data

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

if __name__ == '__main__':
    tickers = ['HD', 'AAPL', 'FB']
    '''
    for ticker in tickers:
        data = extract_from_quandl(ticker)
        data.to_csv('Data/' + ticker + '-quandl.csv')

    print('finished getting')
    '''
    for ticker in tickers:
        data_yahoo = pd.read_csv('Data/' + ticker + '.csv', index_col = 'Date')[['Adj Close', 'Volume']]
        data_yahoo.index = pd.to_datetime(data_yahoo.index)
        data_quandl = pd.read_csv('Data/' + ticker + '-quandl.csv', index_col = 'Date')[['WIKI/' + ticker + ' - Adj. Close', 'WIKI/' + ticker + ' - Adj. Volume']]
        data_quandl.index = pd.to_datetime(data_quandl.index)

        #initializing the figure
        fig = plt.figure(figsize = (16, 8))
        #we are going to have two charts/subplots. first chart will take up 4x4 space in a 5x4 grid
        ax1 = plt.subplot2grid((6,4),(0,0), rowspan = 4, colspan = 4)
        ax1.plot(data_yahoo[['Adj Close']], label = 'yahoo', alpha = 0.5, color = 'r')
        ax1.plot(data_quandl[['WIKI/' + ticker + ' - Adj. Close']], label = 'quandl', alpha = 0.5, color = 'k')
        ax1.legend()
        plt.ylabel('Price')

        #for the second chart, it will take up 1x4 space in a 5x4 grid. it will also share the same axis as the first plot
        ax2 = plt.subplot2grid((6,4),(4,0), sharex = ax1, rowspan = 2, colspan = 4)
        ax2.bar(data_yahoo.index, data_yahoo['Volume'], label = 'yahoo', alpha = 0.5, color = 'r')
        ax2.bar(data_quandl.index, data_quandl['WIKI/' + ticker + ' - Adj. Volume'], label = 'quandl', alpha = 0.5, color = 'k')
        ax2.axes.yaxis.set_ticklabels([]) #for some reason we are hiding the y axis lael for chart 2
        ax2.legend()
        plt.ylabel('Volume')

        #formating
        ax1.grid(True)
        ax2.grid(True)

        plt.xlabel('Date')
        ax2.legend()
        
        #plt.tight_layout()
        plt.subplots_adjust(left = 0.09, bottom = 0.18, right = 0.97, top = 0.94, wspace = 0.20, hspace = 0)
        plt.suptitle(ticker + ' Stock Price')
        plt.setp(ax1.get_xticklabels(),visible = False) #removing the first chart's x label
            
        plt.show()
    
