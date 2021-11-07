# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 21:24:08 2021

@author: lucas
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.colors as cl
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn import metrics
data_cancer = pd.read_csv("D:/Windows/Documentos/AM1/dataR2 (1).csv")

previsores = np.array(data_cancer.iloc[:,0:9].values)
classes = np.array(data_cancer.iloc[:,9:10].values)

x_treino,x_teste,y_treino,y_teste = train_test_split(previsores,classes,test_size = 0.25,
                                                     random_state = 15)

for i in range(0,len(y_treino)):
    if(y_treino[i] == 1 ):
        y_treino[i] = 0
    else:
        y_treino[i] = 1
    

for i in range(0,len(y_teste)):
    if(y_teste[i] == 1 ):
        y_teste[i] = 0
    else:
        y_teste[i] = 1
    

scaler = MinMaxScaler(feature_range=(0,1))

x_treino = scaler.fit_transform(x_treino)
x_teste = scaler.fit_transform(x_teste)

y_treino = np.reshape(y_treino,(y_treino.shape[0],))
y_teste = np.reshape(y_teste ,(y_teste.shape[0],))


c_hp = np.arange(0.25,10.0,0.25)
acuracias = []
revocacao = []
fbs = []
for i in range(0,len(c_hp)):
    clf = svm.SVC(kernel='rbf',C=c_hp[i])
    clf.fit(x_treino, y_treino)

    y_pred = clf.predict(x_teste)
    y_pred = np.reshape(y_pred,(29,1))
    
    acuracias.append(metrics.accuracy_score(y_teste, y_pred))
    revocacao.append(metrics.recall_score(y_teste,y_pred))
    fbs.append(metrics.fbeta_score(y_teste, y_pred,beta=1.5))
    
plt.plot(c_hp,acuracias)
plt.title("Acuracias")
plt.show()

plt.plot(c_hp,revocacao)
plt.title("Revocações")
plt.show()

plt.plot(c_hp,fbs)
plt.title("FBetas")
plt.show()

"""
for i in range(1,len(data_cancer.columns)-1):
    teste = np.transpose(np.array([data_cancer['Age'],data_cancer[data_cancer.columns[i]],data_cancer['Classification']]))
    
    plt.scatter(teste[:,0],teste[:,1],c = teste[:,2], cmap = cl.ListedColormap(['red','blue']))
    plt.xlabel('Age')
    plt.ylabel(data_cancer.columns[i])
    plt.show()
    
"""
