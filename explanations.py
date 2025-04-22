import numpy as np

labels = ['RSI', 'MACD', 'Volatility', 'Volume', 'Return', 'Momentum']

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


def find_smallest_change(point, current_centroid, next_centroid, labels, centroids):
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


