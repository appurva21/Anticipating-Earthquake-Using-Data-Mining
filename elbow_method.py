#elbow method
from sklearn.cluster import KMeans
from pandas.plotting import scatter_matrix
from matplotlib import pyplot as plt


def elbow_method(data):
	#Elbow Method
	#plt.rcParams['figure.figsize'] = (16, 9)
	plt.style.use('ggplot')
	X = data[['Latitude','Longitude','Depth','Magnitude']]
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
