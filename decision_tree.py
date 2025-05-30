import pandas as pd
import numpy as np
from fetch_parse_data import fetch_parse_data
from indicators import *
from scaling import *

def parse_data(df, threshold = 0.3, n = 1):
    df["n_day_return"] = np.zeros(len(df))
    for i in range(len(df)):
        if i + n >= len(df):
            df.iloc[i, df.columns.get_loc("n_day_return")] = -2
            continue
        cur = df.iloc[i]["Close"]
        change = 100 * (df.iloc[i+n]["Close"] - cur) / cur
        if change > threshold:
            df.iloc[i, df.columns.get_loc("n_day_return")] = 1
        elif change < -threshold:
            df.iloc[i, df.columns.get_loc("n_day_return")] = -1
    return df[df["n_day_return"] != -2]

def gini(data):
  counts = data.groupby("n_day_return").size()
  total = sum(counts)
  if total == 0:
    return 0
  sum_of_squares = sum((count/total)**2 for count in counts)
  return 1 - sum_of_squares

def calculate_threshold(data, feature):
  values = np.unique(sorted(data[feature]))
  midpoints = [(values[i] + values[i+1]) / 2 for i in range(len(values)-1)]
  min_gini_split = float('inf')
  threshold = None
  for midpoint in midpoints:
    threshold_true = data[data[feature] <= midpoint]
    threshold_false = data[data[feature] > midpoint]
    if len(threshold_true) == 0 or len(threshold_false) == 0:
      continue
    gini_split = (len(threshold_true) * gini(threshold_true) + len(threshold_false) * gini(threshold_false)) / len(data)
    if gini_split < min_gini_split:
      min_gini_split = gini_split
      threshold = midpoint
  return threshold


def gini_split(data, feature):
  threshold = calculate_threshold(data, feature)
  threshold_true = data[data[feature] <= threshold]
  threshold_false = data[data[feature] > threshold]
  if len(threshold_true) == 0 or len(threshold_false) == 0:
    return float('inf')
  return (len(threshold_true) * gini(threshold_true) + len(threshold_false) * gini(threshold_false)) / len(data)

# MAIN
df = fetch_parse_data()
threshold = 0.003
n_days = 1
df = parse_data(df, threshold, n_days)