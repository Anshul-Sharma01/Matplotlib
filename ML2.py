
# import sys
import csv
import  numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from collections import Counter
# import random
# from typing import List, Any
# from sklearn.metrics import accuracy_score
# from sklearn.model_selection import train_test_split
# from sklearn import datasets
#
# class ScrappyKNN:
#     def __init__(self):
#         self.X_train = None
#         self.y_train = None
#
#     def fit(self, X_train, y_train):
#         self.X_train = X_train
#         self.y_train = y_train
#
#     def predict(self, X_test):
#         predictions = []
#         for row in X_test:
#             label = random.choice(self.y_train)  # Use y_train for labels, not X_train
#             predictions.append(label)
#         return predictions
#
# iris = datasets.load_iris()
#
# X = iris.data
# y = iris.target
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
#
# my_classifier = ScrappyKNN()
#
# my_classifier.fit(X_train, y_train)
#
# predictions = my_classifier.predict(X_test)
#
# print(accuracy_score(y_test, predictions))
# import random
# from typing import List, Any
# from sklearn.metrics import accuracy_score
# from sklearn.model_selection import train_test_split
# from sklearn import datasets
#
# class ScrappyKNN:
#     def __init__(self):
#         self.X_train = None
#         self.y_train = None
#
#     def fit(self, X_train, y_train):
#         self.X_train = X_train
#         self.y_train = y_train
#
#     def predict(self, X_test):
#         predictions = []
#         for row in X_test:
#             label = self.predict_single(row)
#             predictions.append(label)
#         return predictions
#
#     def predict_single(self, x):
#         distances = [self.distance(x, x_train) for x_train in self.X_train]
#         nearest_index = distances.index(min(distances))
#         return self.y_train[nearest_index]
#
#     def distance(self, a, b):
#         return sum((a_i - b_i) ** 2 for a_i, b_i in zip(a, b)) ** 0.5
#
# iris = datasets.load_iris()
#
# X = iris.data
# y = iris.target
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
#
# my_classifier = ScrappyKNN()
#
# my_classifier.fit(X_train, y_train)
#
# predictions = my_classifier.predict(X_test)
#
# print(accuracy_score(y_test, predictions))
# myarr=np.array([[1,2,3,4,5,6]],np.int64)   #first way to create numpy arrays using python objects
# print(myarr[0,1])
# print(myarr.shape)
# print(myarr.dtype)
# myarr[0,1]=45
# print(myarr)
#
# listarray=np.array([[1,2,3,],[5,8,5],[0,3,1]])
# print(listarray)
# print(listarray.size)
#
# dic=np.array({34,23,23})
# print(dic.dtype)
#
# zeros=np.zeros((2,5))
# print(zeros)
# print(zeros.dtype)
# print(zeros.size)
# print(zeros.shape)
#
#
# rng=np.arange(11)
# print(rng)  #creats a numpy array from 0 to n-1
#
# lspace=np.linspace(1,5,10)
# print(lspace)
#
# emp=np.empty((4,6))
# print(emp)
#
# emp_like=np.empty_like(lspace)
# print(emp_like)
#
#
# ide=np.identity(45)
# print(ide)
# print(ide.shape)
#
#
# arr=np.arange(99)
# print(arr)
# print(arr.size)
# print(arr.shape)
#
# arr=arr.reshape(3,33)
# print(arr)
# print(arr.size)
# print(arr.shape)
#
# arr=arr.ravel()
# print(arr)
# print(arr.shape)

# x=[[1,2,3],[4,5,6],[7,1,0]]
# ar=np.array(x)
# print(ar)
#
# print(ar.sum(axis=1))
# print(ar.T)
# print(ar.flat)
# for item in ar.flat:
#     print(item)
#
# print("\n")
# print(ar.ndim)
# print(ar.size)
# print(ar.nbytes)
#
# one=np.array([1,3,4,634,2])
# print(one.argmax())
# print(one.argmin())
# print(one.argsort())
# print(ar.argmax(axis =0))
# print(ar.argmax(axis=1 ))
# print(ar.argsort(axis=0))
# print(ar.reshape(1,9))
#
# ar2=np.array([[1,2,1],[4,0,6],[8,1,0]])
# print(ar+ar2)
#
# print(np.sqrt(ar))
# print(ar.sum())
# # noinspection PyArgumentList
# print(ar.min())
# # noinspection PyArgumentList
# print(ar.max())
#
# print(np.where(ar>5))
# print(type(np.where(ar>5)))
# print(np.count_nonzero(ar))
# print(np.nonzero(ar))
#
# py_ar=[0,4,55,2]
# np_ar=np.array(py_ar)
# print(sys.getsizeof(1)*len(py_ar))
# print(np_ar.itemsize * np_ar.size)
#
# print(np_ar.tolist())

# ************** Matplotlib Tutorial **************

# plt.xkcd()
# plt.style.use('fivethirtyeight')

# Median Developer Salaries by Age
# ages_x = [25,26,27,28,29,30,31,32,33,34,35]
#
# x_indexes=np.arange(len(ages_x))
# width=0.25
#
# dev_y = [38496,42000,46752,49320,53200,56000,62316,64928,67317,68748,73752]
#
# plt.bar(x_indexes-width,dev_y,width=width,color="#444444",label="All Devs")
#
# # Median Python Developer Salaries by Age
# py_dev_y = [45372,48876,53850,57287,63016,65998,70003,70000,71496,75370,83640]
#
# plt.bar(x_indexes,py_dev_y,width=width,color="#5a7d9a",label="Python")
#
#
# # Median JavaScript Developer Salaries by Age
# js_dev_y = [37810,43515,46823,49293,53437,56373,62375,66674,68745,68746,74583]
#
# plt.bar(x_indexes+width,js_dev_y,width=width,color="#adad3b",label="Javascript")
#
#
#
#
# plt.legend()
#
# plt.xticks(ticks=x_indexes,labels=ages_x)
plt.style.use("fivethirtyeight")

data=pd.read_csv("data.csv")
ids=data["Responder_id"]
lang_responses=data["LanguagesWorkedWith"]

language_counter=Counter()
for response in lang_responses:
    language_counter.update(response.split(";"))

languages=[]
popularity=[]

for item in language_counter.most_common(15):
    languages.append(item[0])
    popularity.append(item[1])

# print(languages)
# print(popularity)
#

languages.reverse()
popularity.reverse()

plt.barh(languages,popularity)



plt.title("Most Popular Languages")
# plt.ylabel("Programming Languages")
plt.xlabel("No. of People Who Use")


plt.grid("True")

plt.tight_layout()

plt.savefig("plot.png")

plt.show()






