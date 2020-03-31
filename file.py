#!/usr/bin/env python3.7
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from functions import *
from sklearn.cluster import DBSCAN
import predict as p
# get_directory_layers_from_csv("mnist_512_.csv")

# X contient les valeurs réelles, X_1_0 contient les valeurs réelles X cryptées en 0 et 1
# X,X_1_0,y = readXy("mnist_64_32_16_/mnist_l1_128_l2_64_l3_32_.csv",True)
# X_1 = [X_1_0[i] for i in range(len(X_1_0)) if y[i]==1]
# X_0 = [X_1_0[i] for i in range(len(X_1_0)) if y[i]==0]
# enc_sig_val = encrypting_signature_value(X_1_0)

# encrypting_X_0 = X_to_encrypted_X(X_0,enc_sig_val)
# encrypting_X_1 = X_to_encrypted_X(X_1,enc_sig_val)

# Histogram(encrypting_X_1,"X1cr.png")
# Histogram(encrypting_X_0,"X0cr.png")

#Histogram(X_1,"X1.png")
#Histogram(X_0,"X0.png")
## exemple d’utilisation 
# df,bins=discretise_dataset('mnist_64_32_16_/mnist_l1_64_l2_32_l3_16_.csv',3)
# df is a pandas_core_frame_DataFrame 
# df_X,df_y = pandas_core_frame_DataFrame_to_list(df)
# X_1 = [df_X[i] for i in range(len(df_X)) if df_y[i]=='1']
# X_0 = [df_X[i] for i in range(len(df_X)) if df_y[i]=='0']

# enc_sig_val = encrypting_signature_value(df_X)

# encrypting_X_0 = X_to_encrypted_X(X_0,enc_sig_val)
# encrypting_X_1 = X_to_encrypted_X(X_1,enc_sig_val)
# print(encrypting_X_0)
# Histogram(encrypting_X_1,"mnist1.png")
# Histogram(encrypting_X_0,"mnist0.png")


# bins = [2,3,4]
# hists_files("mnist_512_/mnist_l1_512_.csv",bins)

# layer1,layer2,layer3, y, bins = makes_discretised_Layers("iris_8_10_8_/iris_l1_8_l2_10_l3_8_.csv",10)

# layer1,layer2,layer3,layer4, y, bins = makes_discretised_Layers("makemoons_3_10_10_3_/makemoons_l1_3_l2_10_l3_10_l4_3_.csv",10)

layer1,layer2,layer3, y, bins = makes_discretised_Layers("mnist_64_32_16_/mnist_l1_64_l2_32_l3_16_.csv",10)
# layer1,layer2,layer3, y, bins = makes_discretised_Layers("iris_8_10_8_/iris_l1_8_l2_10_l3_8_.csv",10)

# mat_dist1 = matrice_distances(mnlayer) #layer1 -> mnlayer1


layers = []
layers.append(strTolist(layer1))
layers.append(strTolist(layer2))
layers.append(strTolist(layer3))
# layers.append(strTolist(layer4))

######### DBSCAN

# clusters = clustering(layers,False)

# print(clusters[2].labels_)
# print(clusters[1].labels_)
# print(clusters[0].labels_)
# print(p.dbscan_predict(clusters[2],strTolist(layer3)))

########## KMEANS

clusters,models = p.kmModel(layers,6)
print(clusters[0],y)

# signatures_clusters("mnist_clusters.csv",clusters,y) 

# clustering = DBSCAN(eps=2, min_samples=2,metric='precomputed').fit(mat_dist1)

# l1 , l2, d, dictio = index_columns_and_data_for_percentage_function(clustering.labels_,y3)# y1 -> y3 
# print(classes_percentage_in_clustering(clustering.labels_,y3)) # y1 -> y3