import matplotlib.pyplot as plt
from KMeans import KMeans
from indicators import *
from scaling import *
from fetch_parse_data import fetch_parse_data

df = fetch_parse_data()

s_data = standardise_data(df[labels])
km = KMeans(6, s_data)
km.run()
centroids = km.get_centroids()
clusters = km.get_clusters()

df['Cluster'] = clusters

# Time series with color-coded clusters
plt.figure(figsize=(14, 7))
for cluster in df['Cluster'].unique():
  cluster_data = df[df['Cluster'] == cluster]
  plt.plot(cluster_data.index, cluster_data['Close'], label=f'Cluster {cluster}', marker='o', linestyle='-')
plt.title('Time Series with Cluster Assignments')
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.legend()
plt.grid(True)
plt.show()

# Pie chart showing cluster percentages
cluster_percentages = df['Cluster'].value_counts(normalize=True) * 100
plt.figure(figsize=(6, 6))
plt.pie(cluster_percentages, labels=cluster_percentages.index, autopct='%1.1f%%', startangle=90)
plt.title('Percentage of Time Spent in Each Cluster')
plt.show()

# Cluster transitions
transition_matrix = pd.crosstab(df['Cluster'].shift(1), df['Cluster'])
print("\nCluster Transition Matrix:")
print(transition_matrix)