import pandas as pd
import numpy as np
import yfinance as yf

from indicators import *


def fetch_parse_data(startDate="2014-11-01", endDate="2019-01-01"):
    labels = ['RSI', 'MACD', 'Volatility', 'Volume', 'Return', 'Momentum']

    df = yf.download("SPY", start=startDate, end=endDate, group_by='column')
    df.columns = df.columns.get_level_values(0)

    df['RSI'] = calculate_rsi(df)
    df['MACD'] = calculate_macd(df)
    df['Volatility'] = calculate_volatility(df)
    df['Return'] = calculate_returns(df)
    df['Momentum'] = calculate_momentum(df)

    df = df[df.index >= "2015-01-01"] # to remove any NaN values caused by rolling window indicators
    
    return df