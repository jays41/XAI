import numpy as np
from KMeans import KMeans
from fetch_parse_data import fetch_parse_data
from indicators import *
from scaling import *

# Fetch and prepare data
df = fetch_parse_data()
data = standardise_data(df[labels])

# Evaluate different cluster numbers using elbow method and silhouette analysis
res = []
for i in range(2, 11):  # Test cluster counts from 2 to 10
    print(f"For {i} clusters:")
    
    x = KMeans(i, data)
    x.run()
    
    # Calculate Sum of Squared Errors (SSE) - used for elbow method
    # Lower SSE indicates tighter clusters
    sse = x.get_sse()
    print(f"SSE: {sse}")
    
    # Calculate Silhouette Score - measures cluster separation quality
    # Range: -1 to 1, higher values indicate better defined clusters
    silhouette_score = x.get_silhouette_score()
    print(f"Silhouette score: {silhouette_score}")
    
    # Store results for plotting
    res.append([i, sse, silhouette_score])

print(res)

import matplotlib.pyplot as plt

res_array = np.array(res)

# Extract data for plotting
clusters = res_array[:, 0].astype(int)
sse = res_array[:, 1]
silhouette = res_array[:, 2]

# Create dual-axis plot to compare both metrics
fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot SSE (Elbow Method)
# Look for the "elbow" - point where SSE reduction starts to level off
color = 'tab:red'
ax1.set_xlabel('Number of clusters')
ax1.set_ylabel('SSE', color=color)
ax1.plot(clusters, sse, marker='o', color=color, label='SSE')
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_title('KMeans Evaluation Metrics')

# Plot Silhouette Score on secondary y-axis
# Look for the peak - highest silhouette score indicates optimal clusters
ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('Silhouette Score', color=color)
ax2.plot(clusters, silhouette, marker='o', linestyle='--', color=color, label='Silhouette Score')
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()
plt.grid(True)
plt.show()