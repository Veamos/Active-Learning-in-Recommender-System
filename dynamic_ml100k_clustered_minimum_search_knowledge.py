import os
import sys
import numpy as np
import scipy as sc
import matplotlib
import pandas as pd
import sklearn as sk
import math
from datetime import datetime


from surprise import SVD
from surprise import Dataset
from surprise import accuracy
from surprise import Reader
from surprise.model_selection import train_test_split
from scipy.sparse.linalg import svds
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import collections

df = pd.read_csv('movielens-100k-dataset/ml-100k/udata.csv',sep=";", header=0, engine="python")
user = pd.read_csv('movielens-100k-dataset/ml-100k/uuser.csv',sep=";", header=0, engine ="python")
genre = pd.read_csv('movielens-100k-dataset/ml-100k/ugenre.csv',sep=";", header=0, engine = "python")

# 16 gaps of data

data = []
for i in range(17):
	data.append((i*5000))
data[0] = 1

# A reader is still needed but only the rating_scale param is requiered.

reader = Reader(rating_scale=(1, 5))

# The columns must correspond to user id, item id and ratings (in that order).

# We'll use the famous SVD algorithm.

algo = SVD()

# Active learning 2 - Clustered Knowledge Sampling - Optimize the number of cluster

df_1 = df.pivot(index = 'user', columns ='item', values = 'rating').fillna(0)

df_numpy = df_1.values

np.warnings.filterwarnings('ignore')

df_mask = df_numpy > 0
df_mask = df_mask.astype(int)

rmse_al = []
rmse_ran = []

tstart = datetime.now()

user_ratings_mean = np.mean(df_numpy, axis = 1)
rating_matrix = df_numpy - user_ratings_mean.reshape(-1, 1)

pu, s, qi = svds(rating_matrix, k = 100)

s = np.diag(s)

# Give same importance to all features

mms = MinMaxScaler()
mms.fit(pu)
pu_transformed = mms.transform(pu)

mms = MinMaxScaler()
mms.fit(qi)
qi_transformed = mms.transform(qi)

# Optimize number of cluster

# Sum_of_squared_distances_pu = []
# K = range(1,100)
# for k in K:
# 	km = KMeans(n_clusters=k)
# 	km = km.fit(pu_transformed.T)
# 	Sum_of_squared_distances_pu.append(km.inertia_)

# Sum_of_squared_distances_qi = []
# K = range(1,100)
# for k in K:
#     km = KMeans(n_clusters=k)
#     km = km.fit(qi_transformed.T)
#     Sum_of_squared_distances_qi.append(km.inertia_)

# plt.plot(K, Sum_of_squared_distances_pu, 'bx-')
# plt.xlabel('k')
# plt.ylabel('Sum_of_squared_distances_pu')
# plt.title('Elbow Method For Optimal k')
# plt.show()

# plt.plot(K, Sum_of_squared_distances_qi, 'rx-')
# plt.xlabel('k')
# plt.ylabel('Sum_of_squared_distances_qi')
# plt.title('Elbow Method For Optimal k')
# plt.show()

n_cluster_pu =  44 # Combinaison H-F / J-V / Works
n_cluster_qi = 306 # Combinaison 2 genres

puclusters = KMeans(n_clusters = n_cluster_qi, random_state = 0).fit_predict(pu_transformed)
qiclusters = KMeans(n_clusters = n_cluster_pu, random_state = 0).fit_predict(qi_transformed.T)

# Calcul uinfo and vinfo

numberRow = []
for row in df_numpy:
	i = 0
	for eachValue in row:
		if eachValue == 1 or eachValue == 2 or eachValue == 3 or eachValue == 4 or eachValue == 5:
			i=i+1
	numberRow.append(i)

numberCol = []
for column in df_numpy.T:
	i = 0
	for eachValue in column:
		if eachValue == 1 or eachValue == 2 or eachValue == 3 or eachValue == 4 or eachValue == 5:
			i=i+1
	numberCol.append(i)

for i in range(len(numberRow)):
	numberRow[i]= numberRow[i]/len(numberCol)
for j in range(len(numberCol)):
	numberCol[j] = numberCol[j]/len(numberRow)

# Find the different index of the different cluster

dict_of_index_cluster={}
for i in range(n_cluster_qi):
	dict_of_index_cluster['index'+str(i)] = []
	for j in range(len(puclusters)):
		if(i==puclusters[j]):
			dict_of_index_cluster['index'+str(i)].append(j)

dictCol_of_index_cluster={}
for i in range(n_cluster_pu):
	dictCol_of_index_cluster['indexCol'+str(i)] = []
	for j in range(len(puclusters)):
		if(i==qiclusters[j]):
			dictCol_of_index_cluster['indexCol'+str(i)].append(j)

# Calcul how much I know about each clusters

dict_value={}
for i in range(n_cluster_qi):
	dict_value['index'+str(i)]= 0
	for ind in dict_of_index_cluster['index'+str(i)]:
		dict_value['index'+str(i)] = dict_value['index'+str(i)] + numberRow[ind]

dict_valueCol={}
for i in range(n_cluster_pu):
	dict_valueCol['indexCol'+str(i)]= 0
	for ind in dictCol_of_index_cluster['indexCol'+str(i)]:
		dict_valueCol['indexCol'+str(i)] = dict_valueCol['indexCol'+str(i)] + numberRow[ind]

# Remplace the value for each user / item by the value for the corresponding cluster

for i in range(n_cluster_pu):
	for j in dictCol_of_index_cluster['indexCol'+str(i)]:
		numberCol[j]=dict_valueCol['indexCol'+str(i)]

for i in range(n_cluster_qi):
	for j in dict_of_index_cluster['index'+str(i)]:
		numberRow[j]=dict_value['index'+str(i)]

user = np.array(numberRow)
item = np.array(numberCol)

user = user.reshape(len(user),1)
item = item.reshape(len(item),1)

k = np.dot(user,item.T)
k2 = k.T

knew = pd.DataFrame(k)
knewtranspose = pd.DataFrame(k2)

actualMaxList = knew.max()
actualMinList = knew.min()   # List of all minimums of knew (can be multiple elements)
actualMinListTrans = knewtranspose.min()

maximum = actualMaxList.max()
minimum = actualMinList.min()

trainact = pd.DataFrame(columns=['user','item','rating'])

while minimum < maximum:
	
	miniIndexRow = actualMinListTrans[actualMinListTrans.iloc[:]==minimum].index.tolist()
	miniIndexCol = actualMinList[actualMinList.iloc[:]==minimum].index.tolist()
	for j in miniIndexRow:
		for l in miniIndexCol:
			if df_mask[j][l] == 1:
				trainact = trainact.append(df.loc[(df['user']==(j+1)) & (df['item']==(l+1))],ignore_index = True)
			knew.iloc[j][l] = maximum+1
			knewtranspose.iloc[l][j] = maximum+1
	actualMinList = knew.min()
	actualMinListTrans = knewtranspose.min()
	minimum = actualMinList.min()

trainact = trainact.drop_duplicates(keep='first')

tend = datetime.now()

for i in data:

	algo = SVD()
	algoran = SVD()

	test = df.sample(n = 20000,random_state=1)
	print(test)

	trainact1 = pd.concat([test,trainact]).drop_duplicates(keep=False)
	trainact1 = trainact1.head(i)
	print(trainact1)

	train = pd.concat([df,test]).drop_duplicates(keep=False)
	train = train.sample(n = i)
	print(train)

	trainsetact = Dataset.load_from_df(trainact1[['user', 'item', 'rating']], reader).build_full_trainset()
	trainset = Dataset.load_from_df(train[['user', 'item', 'rating']],reader).build_full_trainset()
	testset = Dataset.load_from_df(test[['user', 'item', 'rating']], reader).build_full_trainset().build_testset()

	algo.fit(trainsetact)
	predictions = algo.test(testset)

	rmse_al.append(accuracy.rmse(predictions, verbose=False))

	algoran.fit(trainset)
	predictionsran = algoran.test(testset)

	rmse_ran.append(accuracy.rmse(predictionsran, verbose=False))

print(rmse_al)
print(" Active learning in ms : ")
print(tend-tstart)
print(rmse_ran)

plt.plot(data,rmse_al,'r')
plt.plot(data,rmse_ran,'b')
plt.axis([0,80000,0.5,2])
plt.title('Clustered Minimum Knowledge Search & Random - RMSE')
plt.xlabel('# data in the trainset')
plt.ylabel('RMSE')
plt.legend(['Active Learning RMSE', 'Random RMSE'], loc='upper left')
plt.show()
    