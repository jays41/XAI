import pandas as pd
import numpy as np
import yfinance as yf

from KMeans import KMeans
from fetch_parse_data import fetch_parse_data
from indicators import *
from scaling import *

def output_centroids(centroids):
  for i in range(len(centroids)):
    print(f"Cluster {i+1}:")
    centroid = unstandardise_point(centroids[i], df)
    for x, y in zip(labels, centroid):
      print(f"{x}: {y:.2f}")
    print()

# MAIN
labels = ['RSI', 'MACD', 'Volatility', 'Volume', 'Return', 'Momentum']

df = fetch_parse_data()

# Get clusters
s_data = standardise_data(df[labels])
km = KMeans(6, s_data)
km.run()
centroids = km.get_centroids()
clusters = km.get_clusters()

df['Cluster'] = clusters

output_centroids(centroids)