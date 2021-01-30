import os
import stockdata as sd
import pandas as pd
import numpy as np
import scipy.optimize as spo
import matplotlib.pyplot as plt
import yfinance as yf

def plot(data, title):
    ax = data.plot(title=title)
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    plt.draw()
    #call plt.show() at the end

def func(allocs, norm_df, start_money, risk_free,):
    #optimizer function
    port_val = sd.make_port(allocs, norm_df, start_money)
    pdr = sd.calc_dr(port_val)
    return sd.sharpe_ratio(pdr, risk_free) * -1 #turns minimizer into maximizer

if __name__ == '__main__':
    #prepare data
    period = '3mo' # String: 1mo, 1yr, see yfinance documentation
    data = sd.fetch_data(period, 'tickers.txt')
    close = sd.get_close(data)
    sd.fill_gaps(close)
    norm = sd.normalize(close)
    risk_free = sd.get_rfr(norm, period)

    #set up optimizer function
    init_val = 1 / len(close.columns)
    x0 = np.full(len(close.columns), init_val)
    cons = {'type': 'eq',
        'fun': lambda x: sum(x) - 1}
    bnds = [(0.05, 0.70) for i in range(0, len(x0))]
    start_money = 28500

    #optimizer
    res = spo.minimize(func, x0, args=(norm, start_money, risk_free), method='SLSQP', bounds=bnds, constraints=cons, options={'disp':True})

    #assigns results to dataframe
    optimized = pd.DataFrame(columns=close.columns)
    optimized.loc[0] = res.x
    optimized = optimized.round(decimals=6)
    optimized.sort_values(by=[0], axis=1, ascending=False, inplace=True)
    print(optimized)
    print(optimized.loc[0] * start_money)

    #display optimized portfolio info and compare with SPY
    opt_port = sd.make_port(optimized.values, norm, start_money)
    sd.port_info(opt_port)
    spy = yf.Ticker('SPY').history(period=period)['Close'].to_frame()
    spynorm = spy / spy.iloc[0]
    sd.fill_gaps(spynorm)
    allspy = sd.make_port([1], spynorm, start_money)
    compare = opt_port.to_frame(name='Optimized').join(allspy.to_frame(name='S&P500'))
    plot(compare, 'Optimum Allocation Value')
    plt.show()

