import pandas as pd
import numpy as np

class KMeans():
  def __init__(self, n, data):
    self.num_clusters = n
    self.data = data
    self.num_features = data.shape[1]
    self.centroids = None
    self.clusters = None
    self.max_iterations = 1000

  def initialise(self):
    min_value = self.data.iloc[:, 0].min()
    max_value = self.data.iloc[:, 0].max()
    self.centroids = np.linspace(min_value, max_value, self.num_clusters)
    self.centroids = np.array([[x] + [0]*(self.num_features-1) for x in self.centroids])
    self.clusters = np.zeros(len(self.data))

  def calculate_distance(self, point, centroid):
    total = 0
    for i, j in zip(point, centroid):
      total += (i - j) ** 2
    return total ** 0.5

  def assign_clusters(self):
    for i in range(len(self.data)):
      values = self.data.iloc[i].values
      distances = []
      for centroid in self.centroids:
        distances.append(self.calculate_distance(values, centroid))
      min_index = distances.index(min(distances))
      self.clusters[i] = min_index

  def calculate_centroids(self):
    for i in range(self.num_clusters):
      cluster_data = self.data[self.clusters == i]
      if len(cluster_data) > 0:
        self.centroids[i] = cluster_data.mean().values
      else:
        self.centroids[i] = np.random.rand(self.num_features)

  def get_sse(self):
    sse = 0
    for i in range(self.num_clusters):
      cluster_data = self.data[self.clusters == i]
      if len(cluster_data) > 0:
        for point in cluster_data.values:
          sse += self.calculate_distance(point, self.centroids[i]) ** 2
    return sse

  def get_cohesion(self, i):
    # Cohesion: The average distance between a point and all other points in the same cluster.
    values = self.data.iloc[i].values
    cluster_data = self.data[self.clusters == self.clusters[i]]
    total = 0
    for point in cluster_data.values:
      total += self.calculate_distance(point, values)
    return total / len(cluster_data)

  def get_separation(self, i):
    # Separation: The average distance between a point and all points in the nearest different cluster.
    values = self.data.iloc[i].values
    distances = []
    for j in range(self.num_clusters):
      if j == self.clusters[i]:
        continue
      distances.append(self.calculate_distance(values, self.centroids[j]))
    cluster_index = distances.index(min(distances))

    cluster_data = self.data[self.clusters == cluster_index]
    total = 0
    for point in cluster_data.values:
      total += self.calculate_distance(point, values)
    return total / len(cluster_data)

  def get_silhouette_score(self):
    score = 0
    for i in range(len(self.data)):
      cohesion = self.get_cohesion(i)
      separation = self.get_separation(i)
      score += (separation - cohesion) / max(cohesion, separation)
    return score / len(self.data)

  def run(self):
    self.initialise()
    prev_centroids = self.centroids
    i = 0
    while i < self.max_iterations:
      self.assign_clusters()
      self.calculate_centroids()
      if np.array_equal(self.centroids, prev_centroids):
        break
      prev_centroids = np.copy(self.centroids)
      i += 1
    
    ret = []
    for row in self.centroids:
      ret.append([float(x) for x in row])
    return ret