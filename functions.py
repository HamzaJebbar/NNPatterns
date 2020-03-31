#!/usr/bin/env python3.7
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
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
    dftemp[0],retbins=pd.cut(dfoneColumn[0], bins=nb_bins, labels=np.arange(nb_bins), right=False,retbins=True)
    df_new=pd.DataFrame(df[0])
    nb_tuples=df.shape[0]
    j=0
    for i in range(1,df.shape[1]):
        df_new[i]=np.copy(dftemp[0][j:j+nb_tuples])
        j+=nb_tuples
    return df_new,retbins

## Lire le fichier contenant les valeurs et les transformer en String
def readXy(filename,toString):
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
        if toString==False:
            X_1_0.append(list(map(lambda x: 1 if x>0 else x ,X[i])))
        else:
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


'''version signature : value'''
def encrypting_signature_value(X_) : # param X_ est X_1_0
    listOfX_ = []
    encrypting_X_ = {}
    value = 1
    for x in range(len(X_)) :
        if not (X_[x] in listOfX_) : 
            listOfX_.append(X_[x])
            encrypting_X_[str(X_[x])] = str(value)
            value += 1
    #print(listOfX_)
    return encrypting_X_ # returns a dict of 'signature : value'


def X_to_encrypted_X(X_,encrypting_signature_value) : # X_ : signatures of a certain class , encrypting_signature_value : the global variable in this code
    listOfX = []
    for x in range(len(X_)) :
        #if not(encrypting_signature_value[str(X_[x])] in listOfX) :
        listOfX.append(encrypting_signature_value[str(X_[x])])
    return listOfX #returns the X_ crypted


def pandas_core_frame_DataFrame_to_list(df) :
    X = []
    y_ = []
    for y in range(len(df)) :
        x = ""
        y_.append(str(df[0][y]))
        for z in range(1,len(df.columns)) :

            x += str(df[z][y])
        X.append(x)
    return X,y_


def hists_files(file,bins) : # "iris_8_10_8_/iris_l1_8_l2_10_l3_8_.csv" should be the file in param
    for x in range(len(bins)) : 
        df,b=discretise_dataset("iris_8_10_8_/iris_l1_8_l2_10_l3_8_.csv",bins[x]) 
        
        name_of_pngHist_class0 = "X_0_disc" + str(bins[x]) + ".png"
        name_of_pngHist_class1 = "X_1_disc" + str(bins[x]) + ".png"

        X_X , y__ = pandas_core_frame_DataFrame_to_list(df) 
        
        X_1_ = [X_X[i] for i in range(len(X_X)) if y__[i]=='1']
        X_0_ = [X_X[i] for i in range(len(X_X)) if y__[i]=='0']

        encr_sig_val_X_X = encrypting_signature_value(X_X)
        enc_X_1_ = X_to_encrypted_X(X_1_,encr_sig_val_X_X)
        enc_X_0_ = X_to_encrypted_X(X_0_,encr_sig_val_X_X)

        Histogram(enc_X_1_,name_of_pngHist_class1)
        Histogram(enc_X_0_,name_of_pngHist_class0)

        

def makes_discretised_Layers(filename,bins) :
    ds,b = discretise_dataset(filename,bins)
    #print("les bins ",b)
    disc_X,y = pandas_core_frame_DataFrame_to_list(ds)

    #print('disc_X\n\n',disc_X,'\n\n')
    if filename[0] == 'i' : # separation par layers de iris
        layer1 = []
        layer2 = []
        layer3 = []
        for x in disc_X :
            layer1.append(x[:8])
            layer2.append(x[8:18])
            layer3.append(x[18:])
        return layer1,layer2,layer3,y,b
    else :
        if filename[0] == 'm' and filename[1]=='a' :
            layer1 = []
            layer2 = []
            layer3 = []
            layer4 = []
            for x in disc_X :
                layer1.append(x[:3])
                layer2.append(x[3:13])
                layer3.append(x[13:23])
                layer4.append(x[23:])
            return layer1,layer2,layer3,layer4,y,b
        else : 
            layer1 = []
            layer2 = []
            layer3 = []
            for x in disc_X :
                layer1.append(x[:64])
                layer2.append(x[64:96])
                layer3.append(x[96:112])
            return layer1,layer2,layer3,y,b

def distance (sig1,sig2) : # special sig1 == sig2
    dist = [[0 for x in range(len(sig1))] for x in range(len(sig2))]
    for x in range(len(sig1)) :
        if x == 0 :
            if sig1[x] == sig2[x] : dist[x][x] = 0
            else : dist[x][x] = 1
        else :
            c = 0 
            while c < x :
                if sig1[x] == sig2[c] : 
                    if c == 0 :
                        dist[c][x] = x
                    else : 
                        dist[c][x] = dist[c-1][x-1] 
                else :
                    if c == 0 :
                        dist[c][x] = min(x , x+1 , dist[c][x-1]) + 1 
                    else :
                        dist[c][x] = min(dist[c-1][x],dist[c][x-1],dist[c-1][x-1]) + 1
                if sig1[c] == sig2[x] : 
                    if c == 0 :
                        dist[x][c] = x
                    else : 
                        dist[x][c] = dist[x-1][c-1]
                else : 
                    if c == 0 :
                        dist[x][c] = min(x , x+1 , dist[x-1][c]) + 1 
                    else :
                        dist[x][c] = min(dist[x-1][c],dist[x][c-1],dist[x-1][c-1]) + 1
                c += 1
            if (sig1[x] == sig2[x]) :
                dist[x][x] = dist[x-1][x-1]
            else : 
                dist[x][x] = min(dist[x][x-1],dist[x-1][x],dist[x-1][x-1]) + 1
    return dist[x][x]

def matrice_distances(layer) :
    matrice = []
    for x in range(len(layer)) : 
        mat =[]
        for y in range(len(layer)) :
            mat.append(distance(layer[x],layer[y]))
        matrice.append(mat)
    return matrice


def layer_sans_doublons(layer) :
    liste = []
    for x in range(len(layer)) :
        if not(layer[x] in liste) : liste.append(layer[x])
    return liste

def index_columns_and_data_for_percentage_function(clustering,y) : #clustering et y doivent etre de la meme longueur 
    liste1 = []
    liste2 = []   
    dictio = {}
    c = 0
    for x in range(len(y)) :
        if not((str(clustering[x])) in liste1) : 
            liste1.append(str(clustering[x]))
            dictio[str(clustering[x])] = c
            c += 1
        if not(("class " + y[x]) in liste2) : 
            liste2.append("class " + y[x])
    d = [0] * len(liste2)
    data = [d] * len(liste1)

    return liste1,liste2,data,dictio


def classes_percentage_in_clustering(clustering,y) :
    #print(y,"\n\n")
    cluster,y_,d,dictio = index_columns_and_data_for_percentage_function(clustering,y)
    #print("cluster ",cluster)
    #print("y_ ",y_)
    #print("dictio ",dictio)
    res = pd.DataFrame(index=cluster,columns=y_,data=d)
    #print(res)
    for x in range(len(y)) : 
        #print(type(dictio[str(clustering[x])]))
        clu = dictio[str(clustering[x])]
        cla = "class " + y[x]
        res[cla][clu] += 1     
    return round(res/len(y) * 100 , 3) 


def strTolist(tab):
    l = []
    for i in range(len(tab)):
        sign = []
        for j in range(len(tab[i])):
            sign.append((int)(tab[i][j]))
        l.append(sign)
    return l

def clustering(layers,lv=True) :
    clustering = []
    for i in range(len(layers)):
        # print("Layer"+str(i))
        if lv==True:
            mat_dist = matrice_distances(layers[i])
            print("Matrice de distance done")
            mat_dist = np.array(mat_dist).astype("float32")
            cluster = DBSCAN(eps=1, min_samples=2,metric='precomputed').fit(mat_dist)
        else:
            cluster = DBSCAN(eps=1, min_samples=2).fit(strTolist(layers[i]))
        print("DBSCAN DONE")
        clustering.append(cluster)
    return clustering
def signatures_clusters(filename,clusters,y) :
	f = open(filename, "w")
	nb_layers = len(clusters)
	for i in range(len(y)) :
		signature = ""+str(y[i])+","
		for j in range(nb_layers) :
			signature += "L"+str(j+1)+":";
			signature += "C"+str(clusters[j].labels_[i])+","
		signature += '\n'
		f.write(signature)
	f.close()