 # Hierarchical Clustering

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3, 4]].values

# Using the dendrogram to find the optimal number of clusters
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X, method = 'ward'))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean distances')
plt.show()

# Fitting Hierarchical Clustering to the dataset
from sklearn.cluster import AgglomerativeClustering
# affinity = euclidean. 
'''
Can be 
- "euclidean"
- "l1"
- "l2"
- "manhattan"
- "cosine"
- "precomputed"

If linkage is "ward", only "euclidean" is accepted
If "precomputed", a distance matrix (instead of a similarity matrix) is needed as input for the fit method.
'''

# linkage = ward: minimizes the variance of the clusters being merged. 
'''
Other options: 
- "ward" minimizes the variance of the clusters being merged. If linkage is "ward", only "euclidean" is accepted.
- "complete" or maximum linkage uses the maximum distances between all observations of the two sets.
- "average" uses the average of the distances of each observation of the two sets.
- "single" uses the minimum of the distances between all observations of the two sets.
'''
hc = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage = 'ward')
y_hc = hc.fit_predict(X)
print("Fitting K-Means to the dataset: ", y_hc)

# Visualising the clusters
plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(X[y_hc == 3, 0], X[y_hc == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(X[y_hc == 4, 0], X[y_hc == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()