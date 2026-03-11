# Unsupervised Learning Example (K-Means Clustering)

import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Sample dataset
X = np.array([[1,2], [1,4], [1,0],
              [10,2], [10,4], [10,0]])

# Create and train model
model = KMeans(n_clusters=2, random_state=0)
model.fit(X)

# Predict clusters
labels = model.labels_
centers = model.cluster_centers_

# Output
print("Cluster Labels:", labels)
print("Cluster Centers:", centers)

# Visualization
plt.scatter(X[:,0], X[:,1], c=labels)
plt.scatter(centers[:,0], centers[:,1], marker='X')
plt.title("Unsupervised Learning - KMeans")
plt.show()
