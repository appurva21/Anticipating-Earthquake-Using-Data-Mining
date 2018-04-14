#Plot

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

def deNormalizeCentroids(data, centroids):
	#denormalize centroid
	max = (data.iloc[:,1:5]).max(axis=0)
	min = (data.iloc[:,1:5]).min(axis=0)
	centroids_df = pd.DataFrame(data = centroids[0:,0:],columns = ['Latitude','Longitude','Depth','Magnitude'])
	denorm_centroids_df = (centroids_df*(max-min))+min
	print(denorm_centroids_df)
	return denorm_centroids_df

def subplot(df,label,centroids, num_clusters):

	plotdata = pd.DataFrame()
	plotdata['Latitude'] = df['Latitude'].values
	plotdata['Longitude'] = df['Longitude'].values
	plotdata['Depth'] = df['Depth'].values
	plotdata['Magnitude'] = df['Magnitude'].values
	plotdata['cluster'] = label
	centroids = deNormalizeCentroids(df, centroids)
	centroids = np.array(centroids)
	c = []
	colors = {0: 'Red', 1: 'Purple', 2: 'Green', 3: 'Black'}
	parameters = ['Latitude', 'Longitude', 'Depth','Magnitude' ]
	count = len(parameters)
	symbol = ['o', '^', 's']
	
	for i in range(count):
		c.append(plotdata[plotdata.cluster == i])
	
	fraction  = int((min(len(c[0].index), len(c[1].index), len(c[2].index)))*0.5)
	
	l = 1
	for i in range(count):
		for j in range(i+1, count):
			plt.subplot(3,2,l)
			for k in range(num_clusters):
				cs = c[k].sample(n=fraction)
				f1 = cs[parameters[i]].values
				f2 = cs[parameters[j]].values
				plt.scatter(f1, f2, c=colors[k], s=7, marker=symbol[k])
				plt.xlabel(parameters[i])
				plt.ylabel(parameters[j])
				plt.title(parameters[j]+" vs "+parameters[i])
			for k in range(num_clusters):
				plt.scatter(centroids[k][i], centroids[k][j], marker=symbol[k], s=50, c=colors[3])
			l+=1
			
	leg = plt.figlegend(['Cluster 0', 'Cluster 1', 'Cluster 2'], loc='upper right', fontsize = 'large')

	# set the linewidth of each legend object
	for legobj in leg.legendHandles:
		legobj.set_linewidth(4.0)
	plt.tight_layout()
	plt.show()
	
def plot(df,label,centroids, num_clusters):

	plotdata = pd.DataFrame()
	plotdata['Latitude'] = df['Latitude'].values
	plotdata['Longitude'] = df['Longitude'].values
	plotdata['Depth'] = df['Depth'].values
	plotdata['Magnitude'] = df['Magnitude'].values
	plotdata['cluster'] = label
	centroids = deNormalizeCentroids(df, centroids)
	centroids = np.array(centroids)
	c = []
	colors = {0: 'Red', 1: 'Purple', 2: 'Green', 3: 'Black'}
	parameters = ['Latitude', 'Longitude', 'Depth','Magnitude' ]
	count = len(parameters)
	symbol = ['o', '^', 's']
	
	for i in range(count):
		c.append(plotdata[plotdata.cluster == i])
	
	fraction  = int((min(len(c[0].index), len(c[1].index), len(c[2].index)))*0.5)
	
	for i in range(count):
		for j in range(i+1, count):
			for k in range(num_clusters):
				cs = c[k].sample(n=fraction)
				f1 = cs[parameters[i]].values
				f2 = cs[parameters[j]].values
				plt.scatter(f1, f2, c=colors[k], s=7, marker=symbol[k])
				plt.xlabel(parameters[i])
				plt.ylabel(parameters[j])
				plt.title(parameters[j]+" vs "+parameters[i])
			for k in range(num_clusters):
				plt.scatter(centroids[k][i], centroids[k][j], marker=symbol[k], s=50, c=colors[3])
			leg = plt.legend(['Cluster 0', 'Cluster 1', 'Cluster 2'], loc='upper right', fontsize = 'large')

			# set the linewidth of each legend object
			for legobj in leg.legendHandles:
				legobj.set_linewidth(4.0)

			plt.tight_layout()
			plt.show()