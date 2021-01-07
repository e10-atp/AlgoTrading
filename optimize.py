import yfinance as yf
import os
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as spo
import math
import numpy as np

def fetch_data(path):
    tickers = ''
    with open(os.path.join(path, 'tickers.txt')) as f:
        for line in f:
            tickers +=  ' {}'.format(line.strip())
    data = yf.download(tickers, period='1y', group_by='ticker', auto_adjust=True)
    return data

def plot(data, title):
    ax = data.plot(title=title)
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    plt.show()

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
    old_date = df.index[0]
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

def sharpe_ratio(port_dr, risk_free):
    #sharpe = E[dr - drfr] / std(dr)
    delta = pd.DataFrame.copy(port_dr)
    delta = delta.sub(risk_free['Close'].values)
    sharpe = delta.mean(axis=0) / delta.std(axis=0)
    k = math.sqrt(252) #square root of data frequency (daily, weekly, monthly, etc.)
    return k * sharpe

def make_port(allocs, norm_df, start_money):
    #returns series containing daily total portfiolio values
    allocated = norm_df.multiply(allocs)
    positions = allocated.multiply(start_money) #starting money
    return positions.sum(axis=1)

def func(allocs, norm_df, start_money, risk_free,):
    #optimizer function
    port_val = make_port(allocs, norm_df, start_money)
    pdr = calc_dr(port_val)
    return sharpe_ratio(pdr, risk_free) * -1 #turns minimizer into maximizer

def port_info(port):
    #financial summary of a portfolio
    print(f'Cumulative Return: {cum_return(port)}')
    pdr = calc_dr(port)
    print(f'Average Daily Return: {pdr.mean()}')
    print(f'Standard Deviation: {pdr.std()}')


if __name__ == '__main__':
    #prepare data
    path = os.getcwd()
    data = fetch_data(path)
    original_data = data
    close = get_close(data)
    fill_gaps(close)
    norm = close / close.iloc[0]
    risk_free = get_rfr(norm)

    #set up optimizer function
    init_val = 1 / len(close.columns)
    x0 = np.full(len(close.columns), init_val)
    cons = {'type': 'eq',
        'fun': lambda x: sum(x) - 1}
    bnds = [(0.00, 1) for i in range(0, len(x0))]
    start_money = 20000

    #optimizer
    res = spo.minimize(func, x0, args=(norm, start_money, risk_free), method='SLSQP', bounds=bnds, constraints=cons, options={'disp':True})

    #assigns results to dataframe
    optimized = pd.DataFrame(columns=close.columns)
    optimized.loc[0] = res.x
    optimized = optimized.round(decimals=6)
    print(optimized.sort_values(by=[0], axis=1, ascending=False))

    #display optimized portfolio info and compare with SPY
    opt_port = make_port(optimized.values, norm, start_money)
    port_info(opt_port)
    spy = yf.Ticker('SPY').history(period='1y')['Close'].to_frame()
    spynorm = spy / spy.iloc[0]
    allspy = make_port([1], spynorm, start_money)
    compare = opt_port.to_frame(name='Optimized').join(allspy.to_frame(name='S&P500'))
    plot(compare, 'Optimum Allocation Value')

