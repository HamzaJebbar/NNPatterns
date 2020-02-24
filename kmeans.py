#!/usr/bin/env python3.7

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import model_selection
from functions import readXy
from functions import Histogram
from sklearn.metrics import accuracy_score
# X,X_1_0,y = readXy("makemoons_3_10_10_3_/makemoons_l1_3_l2_10_l3_10_l4_3_.csv",False)
X,X_1_0,y = readXy("iris_8_10_8_/iris_l1_8_l2_10_l3_8_.csv",False)
# X,X_1_0,y = readXy("mnist_512_/mnist_l1_512.csv",False)
train_X, test_X, train_y, test_y = model_selection.train_test_split(X,y,test_size = 0.2, random_state = 0)
kmeans = KMeans(n_clusters=2, random_state=0).fit(train_X)
#print(len(train_X),len(test_X))
print(kmeans.predict(test_X),test_y)
print(accuracy_score(kmeans.predict(test_X),test_y))
