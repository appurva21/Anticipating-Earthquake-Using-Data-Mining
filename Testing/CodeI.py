from copy import deepcopy
import math
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from pandas.plotting import scatter_matrix
from matplotlib import pyplot as plt
import matplotlib.pyplot as mplpyplot
from orangecontrib.associate.fpgrowth import * 
from mlxtend.preprocessing import OnehotTransactions
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

#plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('ggplot')

# Importing the dataset
data = pd.read_csv('database.csv')
print(data.shape)

year = pd.DataFrame()
year['Date'] = data['Date']
year = np.array(year)
temp = []
#converting date in proper format
for row in year:
	for item in row:
		temp.append([item[:4]])

year = np.asarray(temp)
year = pd.DataFrame(data = year,columns = ['Date'])
data['Date'] = year['Date'].values 


#Elbow Method
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


number_of_clusters = 3

#data_norm = data[['Latitude','Longitude','Depth','Magnitude']]
data_norm = data[['Latitude','Longitude','Depth','Magnitude']].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
f1 = data_norm['Latitude'].values
f2 = data_norm['Longitude'].values
f3 = data_norm['Depth'].values
f4 = data_norm['Magnitude'].values

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
	#print("Random:",C)

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
	print("\nCentroids:")
	print(*C, sep="\n")
	return clusters,C

def deNormalize(dataframe, par, a1, a2, a3, range1, range2):
	list = dataframe[par]
	if par =='Magnitude' or par == 'Depth':
		dataframe[a1] = 0
		dataframe[a2] = 0
		dataframe[a3] = 0
		for j in range(len(list)):
			if list[j]<=range1:
				dataframe.at[j,a1] = 1
			elif list[j]>range1 and list[j]<=range2:
				dataframe.at[j,a2] = 1
			elif list[j]>range2:
				dataframe.at[j,a3] = 1
	elif par == 'Latitude':
		dataframe['NH'] = 0
		dataframe['SH'] = 0
		for j in range(len(list)):
			if list[j]>=0:
				dataframe.at[j,'NH'] = 1
			else:
				dataframe.at[j,'SH'] = 1
				
	elif par == 'Longitude':
		dataframe['Q1'] = 0
		dataframe['Q2'] = 0
		dataframe['Q3'] = 0
		dataframe['Q4'] = 0
		for j in range(len(list)):
			if list[j]>=0 and list[j]<=90:
				dataframe.at[j,'Q1'] = 1
			elif list[j]>90 and list[j]<=180:
				dataframe.at[j,'Q2'] = 1
			elif list[j]>=-90 and list[j]<0:
				dataframe.at[j,'Q4'] = 1
			elif list[j]>=-180 and list[j]<-90:
				dataframe.at[j,'Q3'] = 1
	return dataframe		

	
def plot(df,label,centroids):

	plotdata = pd.DataFrame()
	plotdata['Latitude'] = df['Latitude'].values
	plotdata['Longitude'] = df['Longitude'].values
	plotdata['Depth'] = df['Depth'].values
	plotdata['Magnitude'] = df['Magnitude'].values
	plotdata['cluster'] = label
	#centroids = np.array(cent)
	c = []
	colors = {0: 'Red', 1: 'Purple', 2: 'Green', 3: 'Black'}
	parameters = ['Latitude', 'Longitude', 'Depth','Magnitude' ]
	count = len(parameters)
	symbol = ['o', '^', 's']
	num_clusters = 3
	
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
				plt.scatter(centroids[k][i], centroids[k][j], marker=symbol[k], s=50, c=colors[3])
			l+=1
			
	plt.tight_layout()
	plt.show()

def minsup(x,a,b,c):
	return math.exp(-(a*x)-b) + c

	
labels, centroids = k_Means(number_of_clusters,X)
#print(labels)

"""
#denormalize centroid
max = (data.iloc[:,1:]).max(axis=0)
min = (data.iloc[:,1:]).min(axis=0)
centroids_df = pd.DataFrame(data = centroids[0:,0:],columns = ['Latitude','Longitude','Depth','Magnitude'])
denorm_centroids_df = (centroids_df*(max-min))+min
print(denorm_centroids_df)
"""

#denormalizing data
dn1 = deNormalize(data, "Latitude","", "", "",0,0)
dn2 = deNormalize(dn1, "Longitude","", "", "",0,0)
dn3 = dn2[['NH','SH','Q1','Q2','Q3','Q4']]
dn4 = deNormalize(data_norm, "Depth","DepthLow", "DepthMid", "DepthHigh", 0.10, 0.43)	 #0.10 = 70km , 0.43 = 300km
dn5 = deNormalize(dn4, "Magnitude", "MagnitudeLow", "MagnitudeMid", "MagnitudeHigh",  0.21, 0.55)	#0.21 =5.5mw 0.55= 7.0mw

cluster_map = pd.DataFrame()
#cluster_map['data_index'] = data.index.values
cluster_map['NH'] = dn2['NH'].values
cluster_map['SH'] = dn2['SH'].values
cluster_map['Q1'] = dn2['Q1'].values
cluster_map['Q2'] = dn2['Q2'].values
cluster_map['Q3'] = dn2['Q3'].values
cluster_map['Q4'] = dn2['Q4'].values
cluster_map['DepthLow'] = dn5['DepthLow'].values
cluster_map['DepthMid'] = dn5['DepthMid'].values
cluster_map['DepthHigh'] = dn5['DepthHigh'].values
cluster_map['MagnitudeLow'] = dn5['MagnitudeLow'].values
cluster_map['MagnitudeMid'] = dn5['MagnitudeMid'].values
cluster_map['MagnitudeHigh'] = dn5['MagnitudeHigh'].values
cluster_map['cluster'] = labels


clusters = []

for i in range(number_of_clusters):
		clusters.append(cluster_map[cluster_map.cluster == i])


for i in range(number_of_clusters):
	print("\n----------------------------------Cluster ",i,"(Cluster Size: ", len(clusters[i].index), ")--------------------------------- \n")
	#print(clusters[i].iloc[:,0:-1])

	
#writing clusters to file
cl_filename = ["cluster0.csv", "cluster1.csv", "cluster2.csv"]
for i in range(number_of_clusters):
	clusters[i].to_csv(cl_filename[i], sep=',')


#plotting clusters
#plot(data,labels,centroids)

AR_filename = ["AR_cluster0.csv", "AR_cluster1.csv", "AR_cluster2.csv"]
for i in range(number_of_clusters):
	
	minSupport = minsup(len(clusters[i].index),0.4,0.2,0.6)
	minConfidence =0.7
	#Generating Frequent itemsets
	temp = np.array(clusters[i]).tolist()
	frequent_itemsets = apriori(clusters[i].iloc[:,0:-1], min_support=minSupport, use_colnames=True)
	"""print('\n\nCLUSTER: ',i,'-->Frequent Items:---------')
	print('minSupport: ',minSupport,'\n')
	print(frequent_itemsets)"""

	#Generating Association Rules
	rules=association_rules(frequent_itemsets, metric="confidence", min_threshold=minConfidence)
	"""print('\n\nCLUSTER: ',i,'-->Association Rule:---------')
	print('minConfidence: ',minConfidence,'\n')
	print(rules.iloc[:,0:4])"""
	rules.iloc[:,0:4].to_csv(AR_filename[i], sep=',')

	
	
	
	
"""
#Generating Frequent itemsets
itemsets = list(frequent_itemsets(temp,0.5))
print('\n\nFrequent Items:---------\n')
print('\nOutput Format\n(frozenset(Frequent Itemset), Support Count)\n\n')
print(*itemsets, sep="\n")


#Generating Association Rules
itemsets = dict(frequent_itemsets(temp, .4))
rules =list(association_rules(itemsets, .9))
print('\n\nAsscoiation Rule:---------\n')
print('\nOutput Format\n(frozenset(Antecedent), frozenset(Consequent), Support Count, Confidence)\n\n')
print(*rules, sep="\n")"""

