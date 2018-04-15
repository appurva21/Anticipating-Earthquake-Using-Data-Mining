#encoding

import pandas as pd

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

def encode(data, data_norm, labels):
	dn1 = deNormalize(data, "Latitude","", "", "",0,0)
	dn2 = deNormalize(dn1, "Longitude","", "", "",0,0)
	dn3 = dn2[['NH','SH','Q1','Q2','Q3','Q4']]
	dn4 = deNormalize(data_norm, "Depth","DepthLow", "DepthMid", "DepthHigh", 0.10, 0.43)	 #0.10 = 70km , 0.43 = 300km
	dn5 = deNormalize(dn4, "Magnitude", "MagnitudeLow", "MagnitudeMid", "MagnitudeHigh",  0.21, 0.55)	#0.21 =5.5mw 0.55= 7.0mw

	cluster_map = pd.DataFrame()
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

	return cluster_map

def write(cluster_map, number_of_clusters):
	clusters = []

	for i in range(number_of_clusters):
			clusters.append(cluster_map[cluster_map.cluster == i])


	for i in range(number_of_clusters):
		print("\n----------------------------------Cluster ",i,"(Cluster Size: ", len(clusters[i].index), ")--------------------------------- \n")
		#print(clusters[i].iloc[:,0:-1])

		
	#writing clusters to file
	cl_filename = ["Data_Cluster0.csv", "Data_Cluster1.csv", "Data_Cluster2.csv"]
	for i in range(number_of_clusters):
		clusters[i].to_csv(cl_filename[i], sep=',')
		print('Write Complete for Cluster',i,'. File Name: ',cl_filename[i])	
	return clusters