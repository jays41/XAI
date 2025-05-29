import pandas as pd
import numpy as np
import yfinance as yf

from KMeans import KMeans
from fetch_parse_data import fetch_parse_data
from indicators import *
from scaling import *

def binary_search(point, current_centroid, next_centroid, index, start, end, epsilon=1e-6):
  def substitute(point, index, value):
    new_point = point[:]
    new_point[index] = value
    return new_point

  def get_distance(point, centroid):
    total = 0
    for x, y in zip(point, centroid):
      total += (x - y) ** 2
    return total ** 0.5

  if point[index] == 0:
    return float('inf')

  l = start
  r = end
  while r - l >= epsilon:
    mid = (l + r) / 2
    cur_point = substitute(point, index, mid)
    if get_distance(cur_point, current_centroid) > get_distance(cur_point, next_centroid):
      r = mid
    else:
      l = mid

  new_point = substitute(point, index, r)
  original_point = unstandardise_point(point, df[labels])
  pct_change = (new_point[index] - original_point[index]) / original_point[index]

  return pct_change


def find_smallest_change(point, current_centroid, next_centroid, labels):
  boundary = np.max(np.abs(centroids))
  min_pct = float('inf')
  min_change = None

  changes = []

  for i in range(len(point)):
    cur_min = 0
    positive_change = binary_search(point, current_centroid, next_centroid, i, point[i], point[i]+boundary)
    negative_change = binary_search(point, current_centroid, next_centroid, i, point[i]-boundary, point[i])

    if abs(positive_change) < abs(negative_change):
      cur_min = positive_change
    else:
      cur_min = negative_change

    if abs(cur_min) < abs(min_pct):
      min_pct = cur_min
      min_change = labels[i]

    changes.append([labels[i], cur_min])

  return min_change, min_pct, changes

def explain(date, cluster_labels, centroids, normalised_values):
  point = df.loc[date]
  print(point)
  print(normalised_values)
  cluster_index = int(point['Cluster'])
  print(f"Cluster {cluster_labels[cluster_index]}")
  for i in range(len(centroids)):
    dist = 0
    for x, y in zip(centroids[i], normalised_values):
      dist += (x - y) ** 2
    dist = dist ** 0.5
    print(f"Distance from centroid '{cluster_labels[i]}': {dist:.2f}")

  # find one change that would move this point into an adjacent cluster
  if cluster_index != len(centroids) - 1:
    print(f"In order to move up to the next centroid ({cluster_labels[cluster_index + 1]}):")
    print("Standardised point:", standardise_point(df[labels].loc[date], df))
    print("From:", centroids[cluster_index])
    print("To:", centroids[cluster_index + 1])

    min_change, min_pct, changes = find_smallest_change(standardise_point(df[labels].loc[date], df), centroids[cluster_index], centroids[cluster_index + 1], labels)
    print(f"{'Increase' if min_pct > 0 else 'Decrease'} {min_change} by {(abs(min_pct)*100):.2f}%")
    print(changes)


  # clean up this function


# MAIN
df = fetch_parse_data()
s_data = standardise_data(df[labels])
km = KMeans(6, s_data)
km.run()
centroids = km.get_centroids()
clusters = km.get_clusters()
cluster_labels = ["Crash", "Downtrend", "Correction", "Neutral", "Consolidation", "Bullish Quiet"]
df['Cluster'] = clusters
date = df.index[21]
print(date)
explain(date, cluster_labels, centroids, s_data.loc[date])