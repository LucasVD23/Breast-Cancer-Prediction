# -*- coding: utf-8 -*-
"""
Created on Sat Nov  6 18:08:45 2021

@author: lucas
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as cl


data_cancer = pd.read_csv("D:/Windows/Documentos/AM1/dataR2 (1).csv")

#Idade: atributo numérico discreto, o resto (tirando a classe) é contínuo o resto é ratio
#Atributo Alvo = 1 para paciente saudável, 2 para paciente com câncer

#sumarização
##bins = np.arange(20,100,10)
teste = np.array(data_cancer.iloc[:,0:1].values)
teste_max = teste.max()
teste_min = teste.min()
media = teste.mean()
sd = teste.std()

#Testanto histogramas
for i in data_cancer.columns[0:9]:
    plt.hist(data_cancer[i],density = False)
    plt.title(i)
    plt.show()
#Testando Boxplot
for i in data_cancer.columns[0:9]:
    plt.boxplot(data_cancer[i])
    plt.title(i)
    plt.show()

#testando Relações entre diagnósticos e Parâmetros
for i in data_cancer.columns[0:9]:
    my_dict = {'Negativo': data_cancer[data_cancer['Classification'] == 1][i], 
               'Positivo': data_cancer[data_cancer['Classification'] == 2][i]}
    
    fig, ax = plt.subplots()
    ax.boxplot(my_dict.values())
    ax.set_xticklabels(my_dict.keys())
    plt.title('Relação ' + i + ' X Diagnóstico')
