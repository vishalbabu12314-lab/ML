# Clustering using K-Means

import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Sample data
X = np.array([[2,3], [3,4], [3,3], [5,8], [6,8], [7,7], [8,5]])

# Create model
kmeans = KMeans(n_clusters=2)

# Train model
kmeans.fit(X)

# Cluster labels
labels = kmeans.labels_

# Cluster centers
centers = kmeans.cluster_centers_

print("Cluster Labels:", labels)
print("Cluster Centers:", centers)

# Plot clusters
plt.scatter(X[:,0], X[:,1], c=labels)
plt.scatter(centers[:,0], centers[:,1], marker='X')
plt.title("K-Means Clustering")
plt.show()
