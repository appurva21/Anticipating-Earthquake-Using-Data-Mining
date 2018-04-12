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

data_norm = data[['latitude','longitude','depth','mag']].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
f1 = data_norm['latitude'].values
f2 = data_norm['longitude'].values
f3 = data_norm['depth'].values
f4 = data_norm['mag'].values
"""data_norm = data[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
f1 = data_norm['SepalLengthCm'].values
f2 = data_norm['SepalWidthCm'].values
f3 = data_norm['PetalLengthCm'].values
f4 = data_norm['PetalWidthCm'].values
"""
X = np.array(list(zip(f1, f2, f3, f4)), dtype=np.float32)

def dist(a, b, ax=1):
    return np.linalg.norm(a - b, axis=ax)
	
def k_Means(k, X):

	#coordinates of random centroids
	C_1 = np.random.uniform(0, np.max(X), size=k)
	C_2 = np.random.uniform(0, np.max(X), size=k)
	C_3 = np.random.uniform(0, np.max(X), size=k)
	C_4 = np.random.uniform(0, np.max(X), size=k)
	C = np.array(list(zip(C_1,C_2,C_3,C_4)), dtype=np.float32)
	print("Random:",C)

	C_old = np.zeros(C.shape)
	clusters = np.zeros(len(X))
	error = dist(C, C_old, None)
	while error != 0:
		for i in range(len(X)):
			distances = dist(X[i], C)
			cluster = np.argmin(distances)
			clusters[i] = cluster
		C_old = deepcopy(C)
		for i in range(k):
			points = [X[j] for j in range(len(X)) if clusters[j] == i]
			C[i] = np.mean(points, axis=0)
		error = dist(C, C_old, None)
	print(C)
	return clusters

			

labels = k_Means(4,X)
#print(labels)


cluster_map = pd.DataFrame()
cluster_map['data_index'] = data.index.values
cluster_map['cluster'] = labels

cluster0 = cluster_map[cluster_map.cluster == 0]
cluster1 = cluster_map[cluster_map.cluster == 1]
cluster2 = cluster_map[cluster_map.cluster == 2]
cluster3 = cluster_map[cluster_map.cluster == 3]
print("\n----------------------------------Cluster 0---------------------------------Cluster Size: ",cluster0.shape,"\n")
print(cluster0)
print("\n----------------------------------Cluster 0---------------------------------Cluster Size: ",cluster1.shape,"\n")
print(cluster1)
print("\n----------------------------------Cluster 0---------------------------------Cluster Size: ",cluster2.shape,"\n")
print(cluster2)
print("\n----------------------------------Cluster 0---------------------------------Cluster Size: ",cluster3.shape,"\n")
print(cluster3)
