#Association Rule Generation

import numpy as np
import math
from orangecontrib.associate.fpgrowth import * 
from mlxtend.preprocessing import OnehotTransactions
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

def minsup(x,a,b,c):
	return math.exp(-(a*x)-b) + c

def mine_rules(clusters, number_of_clusters):
	AR_filename = ["Association_Rules_Cluster0.csv", "Association_Rules_Cluster1.csv", "Association_Rules_Cluster2.csv"]
	for i in range(number_of_clusters):
		
		minSupport = minsup(len(clusters[i].index),0.4,0.2,0.6)
		minConfidence =0.7
		#Generating Frequent itemsets
		temp = np.array(clusters[i]).tolist()
		frequent_itemsets = apriori(clusters[i].iloc[:,0:-1], min_support=minSupport, use_colnames=True)
	"""	print('\n\nCLUSTER: ',i,'-->Frequent Items:---------')
		print('minSupport: ',minSupport,'\n')
		print(frequent_itemsets)"""

		#Generating Association Rules
		rules=association_rules(frequent_itemsets, metric="confidence", min_threshold=minConfidence)
	"""	print('\n\nCLUSTER: ',i,'-->Association Rule:---------')
		print('minConfidence: ',minConfidence,'\n')
		print(rules.iloc[:,0:4])"""
		rules.iloc[:,0:4].to_csv(AR_filename[i], sep=',')