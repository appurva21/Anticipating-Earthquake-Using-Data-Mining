#data preprocessing

import numpy as np
import pandas as pd

def get_dataset(filename):
	# Importing the dataset
	data = pd.read_csv(filename)
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
	print("Dataset(Rows,Columns): ",data.shape)
	return data

#normalizing dataset and converting it to np array
def normalize(data):
	data_norm = data[['Latitude','Longitude','Depth','Magnitude']].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
	f1 = data_norm['Latitude'].values
	f2 = data_norm['Longitude'].values
	f3 = data_norm['Depth'].values
	f4 = data_norm['Magnitude'].values

	X = np.array(list(zip(f1, f2, f3, f4)), dtype=np.float32)
	
	return X, data_norm