# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import shuffle


data_cancer = shuffle(pd.read_csv("D:/Windows/Documentos/AM1/dataR2.csv"))

#Idade: atributo numérico discreto, o resto (tirando a classe) é contínuo 
#Atributo Alvo = 1 para paciente saudável, 2 para paciente com câncer

#sumarização
##bins = np.arange(20,100,10)
corrs = data_cancer.corr()['Classification']
teste = np.array(data_cancer.iloc[:,0:1].values)
teste_max = teste.max()
teste_min = teste.min()
media = teste.mean()
sd = teste.std()

for coluna in data_cancer.columns[0:9]:
  print("---------------- " + coluna + " ----------------")
  print("MIN: " + str(data_cancer[coluna].min()))
  print("MAX: " + str(data_cancer[coluna].max()))
  print("MÉDIA: " + str(data_cancer[coluna].mean()))
  print("DESV PADRÃO: "+ str(data_cancer[coluna].std()))
  print("Correlação: "  + str(corrs[coluna]))

#Testanto histogramas
for coluna in data_cancer.columns[0:9]:
    plt.hist(data_cancer[coluna],density = False)
    plt.title(coluna)
    plt.show()
#Testando Boxplot
for coluna in data_cancer.columns[0:9]:
    plt.boxplot(data_cancer[coluna])
    plt.title(coluna)
    plt.show()

#testando Relações entre diagnósticos e Parâmetros
for coluna in data_cancer.columns[0:9]:
    my_dict = {'Negativo': data_cancer[data_cancer['Classification'] == 1][coluna], 
               'Positivo': data_cancer[data_cancer['Classification'] == 2][coluna]}
    
    fig, ax = plt.subplots()
    ax.boxplot(my_dict.values())
    ax.set_xticklabels(my_dict.keys())
    plt.title('Relação ' + coluna + ' X Diagnóstico')
