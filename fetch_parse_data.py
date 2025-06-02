import pandas as pd
import numpy as np
import yfinance as yf
import time

from indicators import *


def fetch_parse_data(sample=False):
    labels = ['RSI', 'MACD', 'Volatility', 'Volume', 'Return', 'Momentum']
    if not sample:
        df = pd.read_csv("data.csv", index_col=0)
    else:
        df = pd.read_csv("sample_data.csv", index_col=0)
        return df
    
    df.columns = df.columns.get_level_values(0)

    df['RSI'] = calculate_rsi(df)
    df['MACD'] = calculate_macd(df)
    df['Volatility'] = calculate_volatility(df)
    df['Return'] = calculate_returns(df)
    df['Momentum'] = calculate_momentum(df)

    df = df[df.index >= "2015-01-01"] # to remove any NaN values caused by rolling window indicators

    return df