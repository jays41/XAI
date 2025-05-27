import pandas as pd
import numpy as np
import copy

class KMeans():
  def __init__(self, n, data):
    self.num_clusters = n
    self.data = data
    self.num_features = data.shape[1]
    self.centroids = None
    self.clusters = None
    self.confidence_scores = []
    self.max_iterations = 1000

  def initialise(self):
    self.centroids = [self.data.sample(1).values.flatten()]
    for _ in range(1, self.num_clusters):
      distances = np.array([np.min([np.linalg.norm(x - c) for c in self.centroids]) for x in self.data.values])
      probabilities = distances ** 2 / np.sum(distances ** 2)
      next_centroid = self.data.iloc[np.random.choice(len(self.data), p=probabilities)].values.flatten()
      self.centroids.append(next_centroid)
    self.clusters = np.zeros(len(self.data))
    self.confidence_scores = np.zeros(len(self.data))

  def calculate_distance(self, point, centroid):
    return np.sqrt(np.sum((point - centroid) ** 2))

  def assign_clusters(self):
    for i, row in enumerate(self.data.values):
      distances = [self.calculate_distance(row, centroid) for centroid in self.centroids]
      self.clusters[i] = np.argmin(distances)

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
      for point in cluster_data.values:
        sse += self.calculate_distance(point, self.centroids[i]) ** 2
    return sse

  def get_cohesion(self, i):
    # Cohesion: The average distance between a point and all other points in the same cluster.
    values = self.data.iloc[i].values
    cluster_data = self.data[self.clusters == self.clusters[i]]
    if len(cluster_data) == 0:
      return 0
    total = sum(self.calculate_distance(point, values) for point in cluster_data.values)
    return total / len(cluster_data)

  def get_separation(self, i):
    # Separation: The average distance between a point and all points in the nearest different cluster.
    values = self.data.iloc[i].values
    own_cluster = int(self.clusters[i])
    min_dist = float('inf')
    nearest_cluster = None
    for j in range(self.num_clusters):
      if j == own_cluster:
        continue
      dist = self.calculate_distance(values, self.centroids[j])
      if dist < min_dist:
        min_dist = dist
        nearest_cluster = j
    cluster_data = self.data[self.clusters == nearest_cluster]
    if len(cluster_data) == 0:
      return 0
    total = sum(self.calculate_distance(point, values) for point in cluster_data.values)
    return total / len(cluster_data)

  def get_silhouette_score(self):
    score = 0
    for i in range(len(self.data)):
      cohesion = self.get_cohesion(i)
      separation = self.get_separation(i)
      if cohesion != 0 and separation != 0:
        current_score = (separation - cohesion) / max(cohesion, separation)
        self.confidence_scores[i] = 100 * (current_score + 1) / 2
        score += current_score
      else:
        self.confidence_scores[i] = 0
    return score / len(self.data)

  def get_confidence_per_cluster(self):
    self.clusters = self.clusters.astype('int')
    totals = np.bincount(self.clusters, weights=self.confidence_scores)
    count_points = np.bincount(self.clusters)
    average_confidence = np.divide(totals, count_points, where=(count_points != 0))
    return {i: val for i, val in enumerate(average_confidence)}

  def run(self):
    self.initialise()
    prev_centroids = copy.deepcopy(self.centroids)
    for _ in range(self.max_iterations):
      self.assign_clusters()
      self.calculate_centroids()
      if all(np.allclose(c1, c2) for c1, c2 in zip(self.centroids, prev_centroids)):
        break
      prev_centroids = copy.deepcopy(self.centroids)

  def get_centroids(self):
    return [centroid.tolist() for centroid in self.centroids]

  def get_clusters(self):
    return self.clusters