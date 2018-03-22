import pandas as pd
import pandas_datareader as pdr
import datetime
import quandl

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
 
    return quandl.get(query_list, 
            returns = 'pandas', 
            start_date = start_date,
            end_date = end_date,
            collapse = 'daily',
            order = 'asc')

if __name__ == '__main__':
    tickers = ['HD','AAPL','FB']
    for ticker in tickers:
        data = extract_from_quandl(ticker)
        data.to_csv('Data/' + ticker + '-quandl.csv')

    print('finished getting')

