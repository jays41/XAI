import pandas as pd
import numpy as np

labels = ['RSI', 'MACD', 'Volatility', 'Volume', 'Return', 'Momentum']

def standardise_data(df):
  means = df.mean()
  stds = df.std()
  standardised_df = (df - means) / stds

  return standardised_df

def standardise_point(point, df):
  means = df[labels].mean().to_numpy()
  stds = df[labels].std().to_numpy()
  point = np.asarray(point)
  standardised = (point - means) / stds
  return standardised.tolist()

def unstandardise_point(point, df):
  means = df[labels].mean().to_numpy()
  stds = df[labels].std().to_numpy()
  point = np.asarray(point)

  return (point * stds + means).tolist()