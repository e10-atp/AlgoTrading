import yfinance as yf
import os
import pandas as pd
import math

def fetch_data(path):
    tickers = ''
    with open(os.path.join(path, 'tickers.txt')) as f:
        for line in f:
            tickers +=  ' {}'.format(line.strip())
    data = yf.download(tickers, period='1y', group_by='ticker', auto_adjust=True)
    return data

def get_close(dataframe):
    #extracts the closing price from larger set of fetched data
    close = pd.DataFrame()
    for ticker, cat in dataframe:
        close[ticker] = dataframe[ticker]['Close']
    return close[~close.index.duplicated(keep='first')] #removes duplicate indexes

def fill_gaps(dataframe):
    #fills NaN values in data
    #forward fill
    dataframe.fillna(method='ffill', inplace=True)
    #backwards fill
    dataframe.fillna(method='bfill', inplace=True)

def normalize(dataframe):
    return dataframe / dataframe.iloc[0]

def calc_dr(dataframe):
    daily = dataframe.copy()
    daily[1:] = (dataframe[1:] / dataframe[:-1].values) - 1
    daily[0:1] = 0
    return daily
    
def cum_return(dataframe):
    return dataframe.iloc[-1] / dataframe.iloc[0] - 1

def get_rfr(df):
    rfr = yf.Ticker('^IRX') #yield received for investing in 13 week T-Bill
    #https://ycharts.com/indicators/3_month_t_bill
    risk_free = rfr.history(period='1y')['Close'].to_frame()
    fill_gaps(risk_free)
    risk_free = risk_free[:] / 100 #comes in percentages

    #fills missing dates in risk_free
    old_date = risk_free.index[0]
    for date in df.index:
        if date not in risk_free.index:
            #print(date)
            risk_free.loc[date, 'Close'] = risk_free.loc[old_date, 'Close']
        old_date = date

    #ensures risk free does not contain indexes not present in df
    for date in risk_free.index:
        if date not in df.index:
            risk_free.drop(date, inplace=True)

    risk_free = risk_free[~risk_free.index.duplicated(keep='first')]
    return risk_free.sort_index()

def make_port(allocs, norm_df, start_money):
    #returns series containing daily total portfiolio values
    allocated = norm_df.multiply(allocs)
    positions = allocated.multiply(start_money) #starting money
    return positions.sum(axis=1)

def sharpe_ratio(port_dr, risk_free):
    #sharpe = E[dr - drfr] / std(dr)
    delta = pd.DataFrame.copy(port_dr)
    delta = delta.sub(risk_free['Close'].values)
    sharpe = delta.mean(axis=0) / delta.std(axis=0)
    k = math.sqrt(252) #square root of data frequency (daily, weekly, monthly, etc.)
    return k * sharpe

def port_info(port):
    #financial summary of a portfolio
    print(f'Cumulative Return: {cum_return(port)}')
    pdr = calc_dr(port)
    print(f'Average Daily Return: {pdr.mean()}')
    print(f'Standard Deviation: {pdr.std()}')