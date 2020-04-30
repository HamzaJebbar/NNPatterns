#!/usr/bin/env python3.7
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import matplotlib
from sklearn.decomposition import PCA
import random as rd

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
    dftemp=dfoneColumn[0]
    # dftemp[0],retbins=pd.cut(dfoneColumn[0], bins=nb_bins, labels=np.arange(nb_bins), right=False,retbins=True)
    # print(retbins)
    df_new=pd.DataFrame(df[0])
    nb_tuples=df.shape[0]
    j=0
    for i in range(1,df.shape[1]):
        df_new[i]=np.copy(dftemp[0][j:j+nb_tuples])
        j+=nb_tuples
    print(df_new)
    return df_new

## Lire le fichier contenant les valeurs et les transformer en String
def readXy(filename):
    f = open(filename, "r")
    matrice = f.read().split('\n')
    y = []
    X = []
    matrice = matrice[1:]
    for i in range(len(matrice)-1) :
        tab = matrice[i].split(',')
        y.append((int)(tab[0]))
        # X.append(list(np.array(tab[1:]).astype("float32")))
        X.append(list(np.array(tab[1:]).astype("float32")))
    return X,y

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

        

def makes_Layers(filename,bins=0,disc=False) :
    # ds = discretise_dataset(filename,bins)
    # #print("les bins ",b)
    # disc_X,y = pandas_core_frame_DataFrame_to_list(ds)
    disc_X,y = readXy(filename)

    #print('disc_X\n\n',disc_X,'\n\n')
    if filename[0] == 'i' : # separation par layers de iris
        layer1 = []
        layer2 = []
        layer3 = []
        '''
        y1 = []
        y2 = []
        y3 = []
        passe = True
        '''
        for x in disc_X :
            layer1.append(x[:8])
            layer2.append(x[8:18]) 
            layer3.append(x[18:])
            '''
            if passe : 
                y1 = y[:8]
                y2 = y[8:18]
                y3 = y[18:]
                passe = False
            '''
        return layer1,layer2,layer3,y
    else :
        if filename[0] == 'm' and filename[1]=='a' :
            layer1 = []
            layer2 = []
            layer3 = []
            layer4 = []
            '''
            y1 = []
            y2 = []
            y3 = []
            y4 = []
            '''
            for x in disc_X :
                layer1.append(x[:3])
                layer2.append(x[3:13])
                layer3.append(x[13:23])
                layer4.append(x[23:])
                '''
                if passe :
                    y1 = y[:3]
                    y2 = y[3:13]
                    y3 = y[13:23]
                    y4 = y[23:]
                    passe = False
                '''
            return layer1,layer2,layer3,layer4,y
        else : 
            layer1 = []
            layer2 = []
            layer3 = []
            #y1 = []
            #y2 = []
            #y3 = []
            for x in disc_X :
                layer1.append(x[:64])
                layer2.append(x[64:96])
                layer3.append(x[96:112])
                '''
                if passe : 
                    y1 = y[:64]
                    y2 = y[64:96]
                    y3 = y[96:112]
                    passe = False
                '''
            return layer1,layer2,layer3,y

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


def sans_doublons(liste) :
    l = []
    for x in range(len(liste)) :
        if not(liste[x] in l) : l.append(liste[x])
    return l


def pourcentages_inter (clusters_of_layer,y) :
    class0 = {} 
    class1 = {}
    clusters = {}
    for i in range(len(y)) :
        #print(type(y[i]))
        if y[i] == 0 : 
            #print("hey ",clusters_of_layer[i])
            if str(clusters_of_layer[i]) in class0 : class0[str(clusters_of_layer[i])] += 1 
            else : class0[str(clusters_of_layer[i])] = 1
        if y[i] == 1 : 
            #print("hey ",clusters_of_layer[i])
            if str(clusters_of_layer[i]) in class1 : class1[str(clusters_of_layer[i])] += 1
            else : class1[str(clusters_of_layer[i])] = 1
          
        if str(clusters_of_layer[i]) in clusters : clusters[str(clusters_of_layer[i])] += 1
        else : clusters[str(clusters_of_layer[i])] = 1
    for key in clusters : 
        if key in class0 : class0[key] = round(class0[key]/clusters[key] *100 , 3)
        if key in class1 : class1[key] = round(class1[key]/clusters[key] * 100 , 3)
    res = [] 
    res.append(class0)
    res.append(class1)
    return res
        
'''
pourcentage .append retour dial la fonction li lfo9 
layers y 
for 3la layers 
pourcentage.append(f)
'''

def pourcentages(clusters , y) :
    res = []
    for x in range(len(clusters)) :
        res.append(pourcentages_inter(clusters[x],y))
    return res


def strTolist(tab):
    l = []
    for i in range(len(tab)):
        sign = []
        for j in range(len(tab[i])):
            sign.append((float)(tab[i][j]))
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

	
def elimination(pourcentages,threshold) :
	clusters_classe0 = []
	clusters_classe1 = []
	for i in range(len(pourcentages)) :
		for j in range(2) :
			clusters = []
			for key in pourcentages[i][j] :
				if(pourcentages[i][j][key]>=threshold) :
					clusters.append(key)
			if(j==0) : clusters_classe0.append(clusters)
			else : clusters_classe1.append(clusters)
	return clusters_classe0, clusters_classe1

def sort_key(dic):
    return dic["source"]
def sort_key_C(dic):
    return dic["target"]

def signatures_clusters2(filename,clusters,clusters_classe0,clusters_classe1,y) :
    tab = []
    nodes = []
    tab_nodes= []
    colors = {0:"blue",1:"red"}
    f = open(filename, "w")
    nb_layers = len(clusters)
    for i in range(len(y)) :
        s = "X"+str(y[i])
        signature = ""+str(y[i])+","
        for j in range(nb_layers) :
            c = "C"+str(j)+str(clusters[j][i])
            k=0
            if s not in tab_nodes:
                tab_nodes.append(s)
                nodes.append({"name":s,"colornode":colors[y[i]],"bordernode":"true"})
            while k < len(tab):
                if tab[k]["source"]==s and tab[k]["target"] == c:
                    tab[k].update({"source":s,"target":c,"value":str(((int)(tab[k]["value"])+1))})
                    break
                k+=1
            if k == len(tab):
                tab.append({"source":s,"target":c,"value":str(1)})

            signature += "L"+str(j+1)+":";
            signature += "C"+str(clusters[j][i])
            signature += "("
            if((str(clusters[j][i]) in clusters_classe0[j]) and (str(clusters[j][i]) in clusters_classe1[j])) : signature += "0,1"
            elif (str(clusters[j][i]) in clusters_classe0[j]) : signature += "0"
            elif (str(clusters[j][i]) in clusters_classe1[j]) : signature += "1"
            signature += "),"
            s = c
        c =  str(y[i])
        k=0
        while k < len(tab):
            if tab[k]["source"]==s and tab[k]["target"] == c:
                tab[k].update({"source":s,"target":c,"value":str(((int)(tab[k]["value"])+1))})
                break
            k+=1
        if k == len(tab):
            tab.append({"source":s,"target":c,"value":str(1)})
        signature += '\n'
        f.write(signature)
    nodes.append({0:colors[0]})
    nodes.append({1:colors[1]})
    f.close()
    return tab,nodes

def keeps_clear_colors(colors) : 
    c = []
    c = [colors[3]]
    c += [colors[7]]
    c += colors[9:18]
    c += colors[19:]
    return c
    # a enlever : 0,2,1,4,5,6,8,18

def gives_color_to_cluster(clusters,clear_colors) :
    dictio = {}
    c = []
    sans_doublons_clusters = sans_doublons(clusters)
    for i in range(len(sans_doublons_clusters)) : 
        dictio[str(sans_doublons_clusters[i])] = clear_colors[i]
    for i in clusters : 
        c.append(dictio[str(i)])
    return c

def gives_colors_to_classes(classes_y) :
    c = []
    for i in classes_y :
        if i == 0 : 
            c.append('orange')
        if i == 1 : 
            c.append('black')
    return c 

def line_points(pca,classes_y) :
    coord = pandas_core_frame_DataFrame_to_list(pca)
    x = coord[1]
    y = coord[0]
    x_ = []
    y_ = []
    for i in range(len(coord[0])):
        x_.append(float(x[i]))
        y_.append(float(y[i]))

    min_x_0 = min_x_1 = max(x_)
    max_x_0 = max_x_1 = min(x_)
    min_y = min(y_)
    max_y = max(y_)

    for i in range(len(x)) : 
        if classes_y[i] == 0 :
            min_x_0 = min(min_x_0,x_[i])
            max_x_0 = max(max_x_0,x_[i])
        if classes_y[i] == 1 :
            min_x_1 = min(min_x_1,x_[i])
            max_x_1 = max(max_x_1,x_[i])
    bi_x = min(max(min_x_0,min_x_1),min(max_x_0,max_x_1))
    bs_x = max(max(min_x_0,min_x_1),min(max_x_0,max_x_1))
    return bi_x,bs_x,min_y,max_y

def plot2D(name,layer,clusters_per_layer,classes_y,layer_num,pca_done=False) :
    if pca_done==False:
        pca = PCA(n_components=2) 
        pca.fit(layer) 
        pca_data = pd.DataFrame(pca.transform(layer))
    else: pca_data = pd.DataFrame(layer)
    clear_colors = keeps_clear_colors(list(matplotlib.colors.cnames.keys()))
    col = gives_color_to_cluster(clusters_per_layer,clear_colors)    
    classes_colors = gives_colors_to_classes(classes_y)
    plt.title('Plot des données')
    plt.xlabel("$x$", fontsize=10)
    plt.ylabel("$y$", fontsize=10)
    plt.scatter(pca_data[0],pca_data[1],s=100,c=col,marker='o',edgecolors=classes_colors)
    
    class0 = plt.scatter([] , [], c='white',marker='o',edgecolors='orange')
    class1 = plt.scatter([] , [] , c='white',marker='o',edgecolors='black')
    plt.legend((class0,class1), ("class 0", "class 1"),loc='upper left')

    bi_x,bs_x,min_y,max_y = line_points(pca_data,classes_y)
    x1 = (bi_x + bs_x)/2
    x2 = (bi_x + bs_x)/4
    plt.plot([min(x1,x2),max(x1,x2)],[min_y,max_y],'k');
    plt.savefig(name+'_layer_'+str(layer_num)+'.png')
    plt.clf()

def plot2D_on_all_layers(name,layers,clusters,classes_y) :
    for i in range(len(layers)) :
        plot2D(name,layers[i],clusters[i],classes_y,i)
