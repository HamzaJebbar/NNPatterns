#!/usr/bin/env python3.7
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#### permet de sauvegarder également un fichier qui contient tous les layers
def get_directory_layers_from_csv(filename):
    tokens=filename.split("_")
    df = pd.read_csv(filename, sep = ',', header = None) 

    
    # creation d'un répertoire pour sauver tous les fichiers
    repertoire=filename[0:-4]
    os.makedirs(repertoire, exist_ok=True)
    string = repertoire+'/'+tokens[0]+'_'
    f=[]
    filenames=[]
    for nb_tokens in range (1,len(tokens)-1):
        name_file=string+'l'+str(nb_tokens)+'_'+tokens[nb_tokens]+'.csv'
        f.append(open(name_file, "w"))
        filenames.append(name_file)
        
        
    # sauvegarde du dataframe dans une chaîne de caracteres
    ch = df.to_string(header=False,
                  index=False,
                  index_names=False).split('\n')
    vals = [','.join(ele.split()) for ele in ch]
    
    # sauvegarde dans des fichiers spécifiques par layer
    token_layer=[]
    token_exemples=[]
    for nb_exemples in range (len(vals)):
        deb=str(df[0][nb_exemples])+','
        # 1 ligne correspond à une chaine
        s=vals[nb_exemples]
        listoftokens=re.findall(r'<b>,(.+?),</b>', s)
        nb_layers=len(listoftokens)
        
        for nb_token in range (nb_layers):
            save_token=''
            save_token=deb+str(listoftokens[nb_token])+'\n'
            
            f[nb_token].write(save_token)

    # sauvegarde d'un fichier qui contient tous les layers en une fois
    # récupération des données pour enlever les <b> et </b>
    df_all=pd.DataFrame()
    myindex=0
    for nb_columns in range(df.shape[1]):
        df[nb_columns]=df[nb_columns].astype(str)
        if (df[nb_columns]!='<b>').all() and (df[nb_columns]!='</b>').all():
            df_all[myindex]=df[nb_columns]
            myindex+=1
    print (df_all.head())
    #cols = [1,2,4,5,12]
    #df_bof=df_all.drop(df_all.columns[cols],axis=1)
    #df_all.drop(df_all.columns[0], axis=1,inplace=True)
    #print (df_all.head())
    # construction du nom du fichier de sauvegarde
    string = repertoire+'/'+tokens[0]+'_'
    for nb_tokens in range (1,len(tokens)-1):
        string+='l'+str(nb_tokens)+'_'+tokens[nb_tokens]+'_'
    string+='.csv'       
    # sauvegarde en .csv
    df_all.to_csv(string, sep=',', encoding='utf-8',index=False)
##### la fonction suivante permet de discretiser en fonction d’une valeur de bin passée en paramètre
def discretise_dataset(filename,bins):
    df = pd.read_csv(filename, sep = ',', header = None) 
    oneColumn = np.array(df[1])
    for i in range(2,df.shape[1]):
        oneColumn=np.append(oneColumn,np.array(df[i]))
    dfoneColumn=pd.DataFrame(oneColumn)
    nb_bins=bins
    dftemp=pd.DataFrame()
    dftemp[0]=pd.cut(dfoneColumn[0], bins=nb_bins, labels=np.arange(nb_bins), right=False)
    df_new=pd.DataFrame(df[0])
    nb_tuples=df.shape[0]
    j=0
    for i in range(1,df.shape[1]):
        df_new[i]=np.copy(dftemp[0][j:j+nb_tuples])
        j+=nb_tuples
    return df_new

## Lire le fichier contenant les valeurs et les transformer en String
def readXy(filename):
    f = open(filename, "r")
    matrice = f.read().split('\n')
    y = []
    X = []
    matrice = matrice[1:]
    for i in range(len(matrice)) :
        tab = matrice[i].split(',')
        y.append((int)(tab[0]))
        # X.append(list(np.array(tab[1:]).astype("float32")))
        X.append(list(np.array(tab[1:]).astype("float32")))

    X_1_0 = []
    for i in range(len(X)) : 
        # X_1_0.append(list(map(lambda x: 1 if x>0 else x ,X[i])))
        x_temp = ""
        for j in range(len(X[i])):
            if X[i][j] > 0:
                x_temp += "1"
            else:
                x_temp += "0"
        X_1_0.append(x_temp)
    return X,X_1_0,y
def Histogram(X,histname):
    fig = plt.hist(X)
    fig = plt.gcf()
    fig.set_size_inches(18.5, 10.5)
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.savefig(histname)
    plt.clf()

#get_directory_layers_from_csv("iris_8_10_8_.csv")

X,X_1_0,y = readXy("iris_8_10_8_/iris_l1_8_l2_10_l3_8_.csv")
X_1 = [X_1_0[i] for i in range(len(X_1_0)) if y[i]==1]
X_0 = [X_1_0[i] for i in range(len(X_1_0)) if y[i]==0]

def listOfX_0(X_0) :
	listOfX_0 = []
	for x in range(len(X_0)) :
		if not (X_0[x] in listOfX_0) :
			listOfX_0.append(X_0[x])
	return listOfX_0	

#print(listOfX_0(X_0)) 
'''
def listOfX_1(X_1) :
	listOfX_1 = []
	for x in range(len(X_1)) : 
		if not (X_1[x] in listOfX_1) :
			listOfX_1.append(X_1[x])
	return listOfX_1

#print(listOfX_1(X_1))


def encryptedX_0(listOfX_0) :
	encrypted_X_0 = {}
	key = 1
	for x in range(len(listOfX_0)) : 
		encrypted_X_0[str(key)] = listOfX_0[x]
		key+=1
	return encrypted_X_0
#print(encryptedX_0(listOfX_0(X_0)))

def encryptedX_1(listOfX_1) :
	encrypted_X_1 = {}
	key = 1
	for x in range(len(listOfX_1)) : 
		encrypted_X_1[str(key)] = listOfX_1[x]
		key+=1
	return encrypted_X_1
#print(encryptedX_1(listOfX_1(X_1)))
'''
def encrypted_X_0_V1(X_0) : 
	listOfX_0 = []
	encrypted_X_0 = {}
	key = 1
	for x in range(len(X_0)) : 
		if not (X_0[x] in listOfX_0) : 
			listOfX_0.append(X_0[x])
			encrypted_X_0[str(key)] = X_0[x]
			key += 1
	return encrypted_X_0
print("encrypted X_0 :\n",encrypted_X_0_V1(X_0))

def encrypted_X_1_V1(X_1) : 
	listOfX_1 = []
	encrypted_X_1 = {}
	key = 1
	for x in range(len(X_1)) : 
		if not (X_1[x] in listOfX_1) : 
			listOfX_1.append(X_1[x])
			encrypted_X_1[str(key)] = X_1[x]
			key += 1
	return encrypted_X_1
print("\nencrypted X_1 :\n",encrypted_X_1_V1(X_1))

encrypted_X_0 = encrypted_X_0_V1(X_0)
encrypted_X_1 = encrypted_X_1_V1(X_1)


Histogram(list(encrypted_X_1_V1(X_1)),"X1.png")
Histogram(list(encrypted_X_0_V1(X_0)),"X0.png")

## exemple d’utilisation 
df=discretise_dataset('iris_8_10_8_/iris_l1_8_l2_10_l3_8_.csv',10)
df.to_csv("iris_8_10_8_/iris_l1_8_l2_10_l3_8_disc.csv", sep=',', encoding='utf-8',index=False)