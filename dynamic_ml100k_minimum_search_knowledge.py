import os
import sys
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
import pandas as pd
import sklearn as sk
import math
import statistics
from datetime import datetime

from surprise import SVD
from surprise import Dataset
from surprise import accuracy
from surprise import Reader
from surprise.model_selection import train_test_split

df = pd.read_csv('movielens-100k-dataset/ml-100k/udata.csv',sep=";", header=0, engine="python")

# 16 gaps of data

data = []
for i in range(17):
	data.append((i*5000))
data[0] = 1

# A reader is still needed but only the rating_scale param is requiered.

reader = Reader(rating_scale=(1, 5))

df_matrix = df.pivot(index = 'user', columns ='item', values = 'rating')

# Transform the dataframe matrix into a numpy matrix

df_numpy = df_matrix.values

np.warnings.filterwarnings('ignore')

df_mask = df_numpy > 0
df_mask = df_mask.astype(int)

# Mean of the mask per row

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

user = np.array(numberRow)
item = np.array(numberCol)

user = user.reshape(len(user),1)
item = item.reshape(len(item),1)

rmse_al = []
rmse_ran = []

##### Active learning 1 - Minimum Knowledge Search #####

# Create score matrix in the dataframe

k = np.dot(user,item.T)
k2 = k.T

knew = pd.DataFrame(k)
knewtranspose = pd.DataFrame(k2)

actualMaxList = knew.max()
actualMinList = knew.min()
actualMinListTrans = knewtranspose.min()

maximum = actualMaxList.max()
minimum = actualMinList.min()

tstart = datetime.now()

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

	test = df.sample(n = 20000)

	trainact1 = pd.concat([test,trainact]).drop_duplicates(keep=False)
	trainact1 = trainact1.head(i)

	train = pd.concat([df,test]).drop_duplicates(keep=False)
	train = train.sample(n = i)

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
plt.title('Minimum Knowledge Search & Random - RMSE')
plt.xlabel('# data in the trainset')
plt.ylabel('RMSE')
plt.legend(['Active Learning RMSE', 'Random RMSE'], loc='upper left')
plt.show()

print(rmse_ran)
print(rmse_al)

