from copy import deepcopy
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('ggplot')

# Importing the dataset
data = pd.read_csv('xclara.csv')

# Getting the values and plotting it
f1 = data['V1'].values
f2 = data['V2'].values
X = np.array(list(zip(f1, f2)))


# k means determine k
distortions = []
# Euclidean Distance Calculator
def dist(a, b, ax=1):
    return np.linalg.norm(a - b, axis=ax)
	
def k_Means(k, X):

		
	# X coordinates of random centroids
	C_x = np.random.randint(0, np.max(X)-20, size=k)
	# Y coordinates of random centroids
	C_y = np.random.randint(0, np.max(X)-20, size=k)
	C = np.array(list(zip(C_x, C_y)), dtype=np.float32)
	#print(C)

	# To store the value of centroids when it updates
	C_old = np.zeros(C.shape)
	# Cluster Labels(0, 1, 2)
	clusters = np.zeros(len(X))
	# Error function - Distance between new centroids and old centroids
	error = dist(C, C_old, None)
	# Loop will run till the error becomes zero
	while error != 0:
		# Assigning each value to its closest cluster
		for i in range(len(X)):
			distances = dist(X[i], C)
			cluster = np.argmin(distances)
			clusters[i] = cluster
		# Storing the old centroid values
		C_old = deepcopy(C)
		# Finding the new centroids by taking the average value
		for i in range(k):
			points = [X[j] for j in range(len(X)) if clusters[j] == i]
			C[i] = np.mean(points, axis=0)
		error = dist(C, C_old, None)

	#calculating distortion
	sse=0
	for i in range(k):
			points = np.array([X[j] for j in range(len(X)) if clusters[j] == i])
			for x in np.nditer(points):
				sse = sse + dist(C[i],x, None)
			
	distortions.append(sse)		
			
def num_of_clusters(X):
	
	K = range(1,6)
	for k in K:
		k_Means(k,X)
	print(distortions)
	# Plot the elbow
	plt.figure(1)
	plt.plot(K, distortions, 'bx-')
	plt.xlabel('k')
	plt.ylabel('Distortion')
	plt.title('The Elbow Method showing the optimal k')
	plt.show()	
	

num_of_clusters(X)
