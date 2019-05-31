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

# Active learning 4 - Attention Based Square(Popularity) * Variance

df_1 = df.pivot(index = 'user', columns ='item', values = 'rating')

df_numpy = df_1.values

np.warnings.filterwarnings('ignore')

df_mask = df_numpy > 0
df_mask = df_mask.astype(int)

rmse_al = []
rmse_ran = []


# Popularity

tstart = datetime.now()

numberCol = []
for column in df_numpy.T:
	i = 0
	for eachValue in column:
		if eachValue == 1 or eachValue == 2 or eachValue == 3 or eachValue == 4 or eachValue == 5:
			i=i+1
	numberCol.append(i)

item_popularity = np.array(numberCol)

item_popularity_normalize = []
for i in range(len(item_popularity)):
	item_popularity_normalize.append((item_popularity[i]-min(item_popularity))/(max(item_popularity)-min(item_popularity)))

# Variance

colVariance = []
for eachCol in df_numpy.T:
	colVariance.append(np.nanvar(eachCol))

item_variance = np.array(colVariance)

# Product

item = np.sqrt(item_popularity_normalize)*item_variance

item = pd.DataFrame(item, columns = ["item"])

# take only the most popular item

item = item.sort_values(by=["item"], ascending = False)
index = item.index.tolist()

trainact = pd.DataFrame(columns=['user','item','rating'])

for i in index:
	trainact = trainact.append(df.loc[df['item']==i],ignore_index = True)

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
plt.axis([0,80000,0.5,1.5])
plt.title('Attention Based Squared Popularity * Variance & Random - RMSE')
plt.xlabel('# data in the trainset')
plt.ylabel('RMSE')
plt.legend(['Active Learning RMSE', 'Random RMSE'], loc='upper left')
plt.show()