import stockdata as sd
import pandas as pd
import mplfinance as mpf
import os

#TO DO: implement buy/sell indicators when automation is in development

def add_bands(df, ticker):
    window = 10
    n = 1.5
    df[ticker, 'EMA'] = df[ticker, 'Close'].ewm(span=window).mean() #exponential moving average
    df[ticker, '+nSD'] = df[ticker, 'EMA'] + df[ticker, 'Close'].rolling(window=window).std() * n
    df[ticker, '-nSD'] = df[ticker, 'EMA'] - df[ticker, 'Close'].rolling(window=window).std() * n

def plot(df, ticker):
    apd = [
    mpf.make_addplot(df['EMA'], color='#1a4bff', width=0.5),
    mpf.make_addplot(df[['+nSD', '-nSD']], color='#7ab4e5', width=0.25)
    ]

    save = dict(fname = os.path.join(os.getcwd(), f'Plots\\{ticker}bands.png'), pad_inches=0.25)

    mpf.plot(df, addplot=apd, type='candle', style='yahoo', volume=True, 
    fill_between=dict(y1=df['-nSD'].values, y2=df['+nSD'].values, alpha=0.2, color='#7ab4e5'),
    title=f'EMA Bollinger Bands for {ticker}',
    ylabel=f'OHLC',
    ylabel_lower='Volume',
    savefig=save)

if __name__ == '__main__':
    period = '3mo'
    data = sd.fetch_data(period, 'tickers.txt')
    sd.fill_gaps(data)

    for ticker in data.stack().columns.values:
        add_bands(data, ticker)
        print(ticker)
        print(data[ticker].iloc[-1].round(decimals=2))
        plot(data[ticker], ticker)

    order = ['Open', 'High', 'Low', 'Close', 'Volume', 'EMA', '-nSD', '+nSD']
    data = data.sort_index(axis=1).reindex(order, axis=1, level=1)