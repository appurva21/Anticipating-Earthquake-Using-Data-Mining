from copy import deepcopy
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from pandas.plotting import scatter_matrix
from matplotlib import pyplot as plt
plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('ggplot')

# Importing the dataset
data = pd.read_csv('query.csv')
print(data.shape)

X = data[['latitude','longitude','depth','mag']]

wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
    
#Plotting the results onto a line graph, allowing us to observe 'The elbow'
plt.xticks(range(1, 11))
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Clusters')
plt.ylabel('WCSS') #within cluster sum of squares
plt.show()


kmeans = KMeans(n_clusters = 4, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
kmeans.fit(X)
print(kmeans.cluster_centers_)


cluster_map = pd.DataFrame()
cluster_map['data_index'] = data.index.values
cluster_map['cluster'] = kmeans.labels_

cluster1 = cluster_map[cluster_map.cluster == 0]
cluster2 = cluster_map[cluster_map.cluster == 1]
cluster3 = cluster_map[cluster_map.cluster == 2]
cluster4 = cluster_map[cluster_map.cluster == 3]
print(cluster1.shape)
print(cluster2.shape)
print(cluster3.shape)
print(cluster4.shape)
