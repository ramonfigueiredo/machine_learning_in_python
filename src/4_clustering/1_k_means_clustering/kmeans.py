# K-Means Clustering

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Mall_Customers.csv')
# Seletion Annual Income (k$) and Spending Score (1-100) columns
X = dataset.iloc[:, [3, 4]].values

# Using the elbow method to find the optimal number of clusters
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11): # 10 clusters
	# k-means++ : selects initial cluster centers for k-mean clustering in a smart way to speed up convergence.
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 0)
    # Fit Annual Income (k$) X Spending Score (1-100)
    # The fit method returns for each observation which cluster it belongs to
    kmeans.fit(X)
    # kmeans.inertia_ = Sum of squared distances of samples to their closest cluster center.
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
# The Elbow method is a heuristic method of interpretation and validation of consistency within cluster analysis designed 
# to help finding the appropriate number of clusters in a dataset.
'''
More precisely, if one plots the percentage of variance explained by the clusters against the number of clusters, 
the first clusters will add much information (explain a lot of variance), but at some point the marginal gain will drop, 
giving an angle in the graph. The number of clusters is chosen at this point, hence the "elbow criterion".
'''
# ==> According to the Elbow Method the best number of cluster is 5
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# Fitting K-Means to the dataset
kmeans = KMeans(n_clusters = 5, init = 'k-means++', random_state = 0)
y_kmeans = kmeans.fit_predict(X)
print("Fitting K-Means to the dataset:", y_kmeans)

# Visualising the clusters
# Cluster 1 has high income and low spending score. A better name for this cluster of clients as "Careful clients" 
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
# Cluster 2 has average income and average spending score. A better name for this cluster of clients as "Standard clients" 
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
# Cluster 3 has high income and high spending score. A better name for this cluster of clients as "Target clients"
# So, cluster 3 is the cluster of clients that would be the main potential target of the mall marketing campaigns
# and it would be very insighful for them all to understand what kind of products are bought by the clients in this cluster 
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
# Cluster 4 has low income and low spending score. A better name for this cluster of clients as "Careless clients" 
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
# Cluster 5 has low income and low spending score. A better name for this cluster of clients as "Sensible clients" 
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()