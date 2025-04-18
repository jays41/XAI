import pandas as pd
import numpy as np
import yfinance as yf

def calculate_rsi(data, window=14):
  delta = data['Close'].diff()
  avg_gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
  avg_loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()

  rs = avg_gain / avg_loss
  rsi = 100 - (100 / (1 + rs))

  return rsi

def calculate_macd(data, fast=12, slow=26, signal=9):
  ema_fast = data['Close'].ewm(span=fast, adjust=False).mean()
  ema_slow = data['Close'].ewm(span=slow, adjust=False).mean()

  macd = ema_fast - ema_slow

  return macd

def calculate_volatility(data, window=21):
  log_return = np.log(data['Close'] / data['Close'].shift(1))
  volatility = log_return.rolling(window=window).std() * np.sqrt(252)

  return volatility

def calculate_returns(data):
  return data['Close'].pct_change()

def calculate_momentum(data, n=10):
  return data['Close'] - data['Close'].shift(n)

# MAIN
labels = ['RSI', 'MACD', 'Volatility', 'Volume', 'Return', 'Momentum']

df = yf.download("SPY", start="2014-11-01", end="2019-01-01", group_by='column')
df.columns = df.columns.get_level_values(0)

df['RSI'] = calculate_rsi(df)
df['MACD'] = calculate_macd(df)
df['Volatility'] = calculate_volatility(df)
df['Return'] = calculate_returns(df)
df['Momentum'] = calculate_momentum(df)

df = df[df.index >= "2015-01-01"] # to remove any NaN values caused by rolling window indicators