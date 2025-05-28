import pandas as pd
import numpy as np

from fetch_parse_data import fetch_parse_data
from KMeans import KMeans
from indicators import *
from scaling import *

def get_percentile(feature, percentile):
  # Assumes percentiles from 5% to 95% in 5% steps → index 0 to 18
  if percentile % 5 != 0 or percentile < 5 or percentile > 95:
    raise ValueError(f"Invalid percentile: {percentile}. Must be a multiple of 5 between 5 and 95.")
  index = (percentile // 5) - 1
  return percentiles[feature][index]

def meets_condition(point, feature, condition):
  idx = labels.index(feature)
  value = float(point[idx])

  if "and" in condition:
    first, second = condition.split("and")
    return meets_condition(point, feature, first) and meets_condition(point, feature, second)

  if "or" in condition:
    first, second = condition.split("or")
    return meets_condition(point, feature, first) or meets_condition(point, feature, second)

  if "th %ile" in condition:
    x = condition.index("th %ile")
    new_condition = condition[:x]
    if "to" in new_condition:
      low, high = new_condition.split("to")
      # print(get_percentile(feature, int(low)) <= value <= get_percentile(feature, int(high)))
      return get_percentile(feature, int(low)) <= value <= get_percentile(feature, int(high))
    else:
      threshold = get_percentile(feature, int(condition[x-2:x]))
      if ">=" in condition:
        # print(value >= threshold)
        return value >= threshold
      elif "<=" in condition:
        # print(value <= threshold)
        return value <= threshold
      elif ">" in condition:
        # print(value > threshold)
        return value > threshold
      elif "<" in condition:
        # print(value < threshold)
        return value < threshold

  if ">=" in condition:
    # print(f"{value} in {condition} -> condition value |{value >= float(condition.split('>=')[1])}|")
    return value >= float(condition.split(">=")[1])
  elif "<=" in condition:
    # print(f"{value} in {condition} -> condition value |{value <= float(condition.split('<=')[1])}|")
    return value <= float(condition.split("<=")[1])
  elif ">" in condition:
    # print(f"{value} in {condition} -> condition value |{value > float(condition.split('>')[1])}|")
    return value > float(condition.split(">")[1])
  elif "<" in condition:
    # print(f"{value} in {condition} -> condition value |{value < float(condition.split('<')[1])}|")
    return value < float(condition.split("<")[1])
  elif "to" in condition:
    low, high = condition.split("to")
    # print(f"{value} in {condition}: {float(low) <= value <= float(high)}")
    return float(low) <= value <= float(high)
  else:
    print(f"Condition not accounted for: {condition}")
    return False

def classify(point):
  point = np.asarray(point)
  tree = pd.read_csv("rules.csv")
  tree.set_index("Node", inplace=True)

  # rsi, macd, vol, volume, ret, momentum = point

  path = []

  if float(point[-1]) > 0.5:
    cur_index = 0
  elif float(point[-1]) < -0.5:
    cur_index = 9
  else:
    cur_index = 18

  while True:
    path.append(cur_index)
    node = tree.loc[cur_index]

    # print(f"{cur_index}: {node['Scenario']}")

    if node["Children"] == "-":
      return node["Scenario"], path

    next_node = None

    children = [int(child) for child in node["Children"].split("|")]
    for child in children:
      child_node = tree.loc[child]
      if str(cur_index) not in str(child_node["Parent"]):
        continue
      # print(f"Trying child node {child} condition: {child_node['Condition']}")
      if meets_condition(point, child_node["Feature"], child_node["Condition"]):
        next_node = child_node
        cur_index = child
        break

    if next_node is None:
      return node["Scenario"], path
    else:
      node = next_node

  return "Unclassified"


# get confidence score as how similar the paths of the points in each centroid are to the path of the point itself
def get_confidence_scores():
  def path_similarity(cur_path, centroid_path):
    if len(cur_path) == 0 or len(centroid_path) == 0:
      return 0

    if cur_path[0] != centroid_path[0]:
      return 0

    return 1 + path_similarity(cur_path[1:], centroid_path[1:])

  res = {}
  x = [unstandardise_point(centroid, df) for centroid in centroids]
  data = standardise_data(df[labels])
  for cluster_idx in range(len(centroids)):
    confidence_score = 0
    centroid_path = classify(x[cluster_idx])[1]
    path_length = len(centroid_path)
    for i in range(len(df[df["Cluster"] == cluster_idx])):
      cur_path = classify(data.iloc[i])[1]
      confidence_score += path_similarity(cur_path, centroid_path) / path_length
    percentage_confidence_score = 100 * confidence_score / len(df[df['Cluster'] == cluster_idx])
    res[cluster_idx] = percentage_confidence_score
  
  return res


# MAIN
df = fetch_parse_data()
s_data = standardise_data(df[labels])
km = KMeans(6, s_data)
km.run()
centroids = km.get_centroids()
clusters = km.get_clusters()
df['Cluster'] = clusters

# Define percentiles every 5% from 5 to 95
percentile_values = np.arange(5, 100, 5)

# Compute percentiles for each feature
percentiles = {
    label: np.percentile(df[label].values, percentile_values)
    for label in labels
}

confidence_scores = get_confidence_scores()

i = 0
x = [unstandardise_point(centroid, df) for centroid in centroids]
res = []
for c in x:
  print(c)
  ret, path = classify(c)
  print(ret, path)
  res.append([i, c, ret])
  print("---------------")