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

class Leaf:
  def __init__(self, label, counts):
    self.label = label
    self.counts = counts

  def __str__(self):
    return f"Leaf: {self.label}, Counts: {self.counts}"

class Node:
  def __init__(self, feature, threshold, left, right, counts):
    self.feature = feature
    self.threshold = threshold
    self.left = left
    self.right = right
    self.counts = counts

  def __str__(self):
    return f"Node: Feature -> {self.feature}, Threshold -> {self.threshold}, Counts: {self.counts}"

class DecisionTree:
  def __init__(self, max_depth=float('inf')):
    self.MAX_DEPTH = max_depth
    self.root = None
  
  def gini(self, data):
    counts = data.groupby("n_day_return").size()
    total = sum(counts)
    if total == 0:
      return 0
    sum_of_squares = sum((count/total)**2 for count in counts)
    return 1 - sum_of_squares

  def calculate_threshold(self, data, feature):
    values = np.unique(sorted(data[feature]))
    midpoints = [(values[i] + values[i+1]) / 2 for i in range(len(values)-1)]
    min_gini_split = float('inf')
    threshold = None
    for midpoint in midpoints:
      threshold_true = data[data[feature] <= midpoint]
      threshold_false = data[data[feature] > midpoint]
      if len(threshold_true) == 0 or len(threshold_false) == 0:
        continue
      gini_split = (len(threshold_true) * self.gini(threshold_true) + len(threshold_false) * self.gini(threshold_false)) / len(data)
      if gini_split < min_gini_split:
        min_gini_split = gini_split
        threshold = midpoint
    return threshold

  def gini_split_datasets(self, data, feature):
    threshold = self.calculate_threshold(data, feature)
    threshold_true = data[data[feature] <= threshold]
    threshold_false = data[data[feature] > threshold]
    return threshold_true, threshold_false

  def gini_split_value(self, data, feature, threshold=None):
    if not threshold:
      threshold_true, threshold_false = self.gini_split_datasets(data, feature)
    else:
      threshold_true = data[data[feature] <= threshold]
      threshold_false = data[data[feature] > threshold]
    if len(threshold_true) == 0 or len(threshold_false) == 0:
      return float('inf')
    return (len(threshold_true) * self.gini(threshold_true) + len(threshold_false) * self.gini(threshold_false)) / len(data)
  
  def build_tree(self, data, depth):
    if len(data) == 0:
      return None

    counts = data.groupby("n_day_return").size()

    if sum(counts) == counts.idxmax(): # if there is only one class
      return Leaf(counts.idxmax(), counts.to_dict())

    if depth > self.MAX_DEPTH:
      return Leaf(counts.idxmax(), counts.to_dict())

    best_feature = None
    best_threshold = None
    min_gini_split = float('inf')

    for feature in labels:
      threshold = self.calculate_threshold(data, feature)
      gini_split = self.gini_split_value(data, feature, threshold)
      if gini_split < min_gini_split:
        min_gini_split = gini_split
        best_feature = feature
        best_threshold = threshold

    if best_feature is None:
      return Leaf(counts.idxmax(), counts.to_dict())

    threshold_true = data[data[best_feature] <= best_threshold]
    threshold_false = data[data[best_feature] > best_threshold]

    left_subtree = self.build_tree(threshold_true, depth + 1)
    right_subtree = self.build_tree(threshold_false, depth + 1)

    return Node(feature=best_feature, threshold=threshold, left=left_subtree, right=right_subtree, counts=counts.to_dict())
  
  def train(self, df):
    self.root = self.build_tree(df, 0)

  def print_tree(self, node, spacing=""):
    print(spacing + str(node))

    if isinstance(node, Leaf):
        return

    print(spacing + "+-- True:")
    self.print_tree(node.left, spacing + "|   ")

    print(spacing + "+-- False:")
    self.print_tree(node.right, spacing + "    ")
  
  def output(self):
    if self.root is None:
      return "Call the method train() before output()"
    
    return self.print_tree(self.root)


# MAIN
df = fetch_parse_data()
threshold = 0.003
n_days = 1
df = parse_data(df, threshold, n_days)

dt = DecisionTree()
dt.train(df)
dt.output()