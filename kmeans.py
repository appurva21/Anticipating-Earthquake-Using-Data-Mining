#kmeans
import numpy as np
import pandas as pd
from copy import deepcopy

def dist(a, b, ax=1):
    return np.linalg.norm(a - b, axis=ax)
	
def k_Means(k, X):

	#coordinates of random centroids
	C_1 = np.random.uniform(0, np.max(X), size=k)
	C_2 = np.random.uniform(0, np.max(X), size=k)
	C_3 = np.random.uniform(0, np.max(X), size=k)
	C_4 = np.random.uniform(0, np.max(X), size=k)
	C = np.array(list(zip(C_1,C_2,C_3,C_4)), dtype=np.float32)

	C_old = np.zeros(C.shape)
	clusters = np.zeros(len(X))
	error = dist(C, C_old, None)
	while error != 0:
		for i in range(len(X)):
			distances = dist(X[i], C)
			cluster = np.argmin(distances)
			clusters[i] = int(cluster)
		C_old = deepcopy(C)
		for i in range(k):
			points = [X[j] for j in range(len(X)) if clusters[j] == i]
			C[i] = np.mean(points, axis=0)
		error = dist(C, C_old, None)
	c_df = pd.DataFrame(data = C[0:,0:],columns = ['Latitude','Longitude','Depth','Magnitude'])
	print('\n[Normalized Centroids] for 3 Clusters: \n',c_df)
	#print(*C, sep="\n")
	return clusters,C

