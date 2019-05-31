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

# We'll use the famous SVD algorithm.

algo = SVD()

# Active learning 3 - Uncertainty Reduction Variance

df_1 = df.pivot(index = 'user', columns ='item', values = 'rating').fillna(0)

df_numpy = df_1.values

np.warnings.filterwarnings('ignore')

df_mask = df_numpy > 0
df_mask = df_mask.astype(int)

rmse_al = []
rmse_ran = []

# Calcul variance for each user/item

tstart = datetime.now()

colVariance = []
for eachCol in df_numpy.T:
	colVariance.append(np.nanvar(eachCol))

item = np.array(colVariance)

item = item.reshape(len(item),1)

var = pd.DataFrame(item, columns=['variance'])

# # Find the maximum - most incertain ratings about item

maximum = float(var.max())
minimum = float(var.min())

trainact = pd.DataFrame(columns=['user','item','rating'])

while maximum > minimum:
	ind = int(var.idxmax())
	trainact = trainact.append(df.loc[df['item']==(ind+1)],ignore_index = True)
	var = var.drop(index = ind)
	maximum = float(var.max())

trainact = trainact.drop_duplicates()

tend = datetime.now()

for i in data:

	algo = SVD()
	algoran = SVD()

	test = df.sample(n = 20000,random_state=1)

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
plt.title('Uncertainty Reduction Variance & Random - RMSE')
plt.xlabel('# data in the trainset')
plt.ylabel('RMSE')
plt.legend(['Active Learning RMSE', 'Random RMSE'], loc='upper left')
plt.show()