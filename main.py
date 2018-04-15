#main
import numpy as np
from data_preprocess import get_dataset, normalize
from elbow_method import elbow_method
from kmeans import k_Means
from encoding import encode,write
from association_rule_mining import mine_rules
from plot import plot, subplot

#filename of dataset
filename = 'final_dataset.csv'

#reading data
data = get_dataset(filename)

#plotting wcss vs number of clusters
elbow_method(data)

#number of clusters
number_of_clusters = 3

#normalizing data
X, data_norm = normalize(data)

#kmeans++
labels, centroids = k_Means(number_of_clusters,X)

subplot(data,labels,centroids, number_of_clusters)

#encoding attributes
cluster_map = encode(data, data_norm, labels)

#writing clusters to file
clusters = write(cluster_map, number_of_clusters)

#mining association rules
mine_rules(clusters, number_of_clusters)