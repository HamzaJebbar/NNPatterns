#!/usr/bin/env python3.7
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from functions import *
#get_directory_layers_from_csv("iris_8_10_8_.csv")

X,X_1_0,y = functions.readXy("iris_8_10_8_/iris_l1_8_l2_10_l3_8_.csv",True)
X_1 = [X_1_0[i] for i in range(len(X_1_0)) if y[i]==1]
X_0 = [X_1_0[i] for i in range(len(X_1_0)) if y[i]==0]
#print("X_0 : \n",X_0)
#print("\nX_1 :\n",X_1)


encrypting_X_0 = encrypting(X_0)
encrypting_X_1 = encrypting(X_1)
printN("encrypting X_0 :\n", encrypting_X_0,"\nencrypting X_1 : \n", encrypting_X_1)


encrypted_X_0 = X_to_encrypted_X(X_0,encrypting_X_0)
encrypted_X_1 = X_to_encrypted_X(X_1,encrypting_X_1)
# print("encrypted X_0\n",encrypted_X_0)
# print("encrypted_X_1\n",encrypted_X_1)

Histogram(encrypted_X_1,"X1cr.png")
Histogram(encrypted_X_0,"X0cr.png")

#Histogram(X_1,"X1.png")
#Histogram(X_0,"X0.png")
## exemple d’utilisation 
df=discretise_dataset('iris_8_10_8_/iris_l1_8_l2_10_l3_8_.csv',8)
df.to_csv("iris_8_10_8_/iris_l1_8_l2_10_l3_8_disc.csv", sep=',', encoding='utf-8',index=False)