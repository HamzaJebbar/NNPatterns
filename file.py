#!/usr/bin/env python3.7
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from functions import *
from sklearn.cluster import DBSCAN

# get_directory_layers_from_csv("mnist_512_.csv")

# X contient les valeurs réelles, X_1_0 contient les valeurs réelles X cryptées en 0 et 1
X,X_1_0,y = readXy("mnist_512_/mnist_l1_512_.csv",True)
X_1 = [X_1_0[i] for i in range(len(X_1_0)) if y[i]==1]
X_0 = [X_1_0[i] for i in range(len(X_1_0)) if y[i]==0]
enc_sig_val = encrypting_signature_value(X_1_0)

encrypting_X_0 = X_to_encrypted_X(X_0,enc_sig_val)
encrypting_X_1 = X_to_encrypted_X(X_1,enc_sig_val)
# print(encrypting_X_0)
# print(encrypting_X_1)

Histogram(encrypting_X_1,"X1cr.png")
Histogram(encrypting_X_0,"X0cr.png")

#Histogram(X_1,"X1.png")
#Histogram(X_0,"X0.png")
## exemple d’utilisation 
# df=discretise_dataset('mnist_512_/mnist_l1_512.csv',8)
# df.to_csv("iris_8_10_8_/iris_l1_8_l2_10_l3_8_disc.csv", sep=',', encoding='utf-8',index=False)
# df is a pandas_core_frame_DataFrame 

# df_X,df_y = pandas_core_frame_DataFrame_to_list(df)

bins = [2,3,4]
hists_files("mnist_512_/mnist_l1_512_.csv",bins)

layer1,layer2,layer3 = makes_discretised_Layers("iris_8_10_8_/iris_l1_8_l2_10_l3_8_.csv",10)
'''
print("layers : \n\n",len(layer1[0]),"\n\n",len(layer2[0]),"\n\n",len(layer3[0]))
'''

malayer1,malayer2,malayer3,malayer4 = makes_discretised_Layers("makemoons_3_10_10_3_/makemoons_l1_3_l2_10_l3_10_l4_3_.csv",10)
'''
print("layers : \n\n",len(malayer1[0]),"\n\n",len(malayer2[0]),"\n\n",len(malayer3[0]),"\n\n",len(malayer4[0]))
'''

mnlayer = makes_discretised_Layers("mnist_512_/mnist_l1_512_.csv",10)
'''
print("layers : \n\n", len(mnlayer[0]))
'''
#print(layer1,"\n\n")
layer1_sans_doublons = layer_sans_doublons(layer1)
#print(layer1_sans_doublons,"\n\n")

#print(matrice_distances(layer1_sans_doublons))

print(layer1_sans_doublons)

mat_dist = matrice_distances(layer1_sans_doublons)

#print(mat_dist)

mat_dist = np.array(mat_dist).astype("float32")

clustering = DBSCAN(eps=2, min_samples=2,metric='precomputed').fit(mat_dist)






print(clustering.labels_)
