#main
import numpy as np
from data_preprocess import get_dataset, normalize
from elbow_method import elbow_method
from kmeans import k_Means
from encoding import encode,write
from association_rule_mining import mine_rules
from plot import plot, subplot

filename = 'final_dataset.csv'
number_of_clusters = 3

data = get_dataset(filename)

elbow_method(data)

X, data_norm = normalize(data)

labels, centroids = k_Means(number_of_clusters,X)

subplot(data,labels,centroids, number_of_clusters)

cluster_map = encode(data, data_norm, labels)
clusters = write(cluster_map, number_of_clusters)

mine_rules(clusters, number_of_clusters)