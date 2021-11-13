# -*- coding: utf-8 -*-

#AVISO: A EXECUÇÃO DESTES TESTES TENDE A DEMORAR POUCO MAIS DE UMA HORA
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest,chi2
from sklearn.utils import shuffle
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn import svm
from sklearn import metrics


data_cancer = shuffle(pd.read_csv("dataR2.csv"),random_state = 15)

#Classificadores usados
dtc = DecisionTreeClassifier()
rfc = RandomForestClassifier()
knnc = KNeighborsClassifier()
svmc = svm.SVC()
adac = AdaBoostClassifier()

#Parâmetros a serem testados
dtc_params = {'criterion': ['entropy','gini'],
              'max_depth': range(2,30,2),
              'min_samples_leaf':range(2,10,2)}

rfc_params = {'criterion': ['entropy','gini'],
              'max_depth': range(2,30,2),
              'min_samples_leaf':range(2,10,2),
              'bootstrap':[False,True],
              }
knnc_params = {'n_neighbors': range(2,15,1),
               'weights' : ['uniform', 'distance'],
               'metric':['minkowski','euclidean','manhattan']}
svmc_params = {'kernel' : ['linear','rbf','poly'],
               'C': np.arange(0.25,2,0.01)}
adac_params = {'n_estimators' : range(1,200,5),
               'learning_rate': np.arange(0.1,2,0.1)}

#dicionarios referenciando classificadores e parâmetros
classifiers = {'Decision Tree': dtc, 'RandomForest':rfc,'KNN':knnc,'SVM':svmc, 'AdaBoost':adac}
parametros = {'Decision Tree': dtc_params, 'RandomForest': rfc_params, 'KNN':knnc_params, 'SVM':svmc_params, 'AdaBoost': adac_params}
#criação da métrica de Fbeta com b = 1.5
fb_scorer = metrics.make_scorer(metrics.fbeta_score,beta=1.5)

#k_fold com 5 splits
kf = KFold(n_splits=5,shuffle = False)

melhor_cf_por_k = dtc
melhor_score = 0
melhor_conj_atributos = []
melhores_metricas = {}
conjuntos_atts =[]

#Testando valores de K de 2 a 9 atributos
for i in range(2,10):
    atr_selector = SelectKBest(score_func=chi2,k=i).fit(data_cancer[data_cancer.columns[0:9]],data_cancer[data_cancer.columns[9:10]])
    atts = []
    atributos_bool = atr_selector.get_support(indices = False)
    for j in range(0,len(atributos_bool)):
        if(atributos_bool[j] == True):
            atts.append(data_cancer.columns[j])
    conjuntos_atts.append(atts)
    
    previsores = np.array(data_cancer.iloc[:,0:9].values)
    classes = np.array(data_cancer.iloc[:,9:10].values)
    previsores = atr_selector.fit_transform(previsores,classes)
    
    scaler = MinMaxScaler(feature_range=(0,1))
    previsores = scaler.fit_transform(previsores)
    classes = np.reshape(classes, (classes.shape[0],))
    
    #Transformando classes, dignóstico negativo = 0 e positivo = 1
    for j in range(0,len(classes)):
        if(classes[j] == 1 ):
            classes[j] = 0
        else:
            classes[j] = 1
    
    #dicionário que guarda os classificadores com seus melhores conjuntos de hiperparâmetros
    best = {}
    for j in classifiers:
        gs = GridSearchCV(classifiers[j],param_grid=parametros[j],scoring = fb_scorer,cv=kf.get_n_splits(previsores))
        gs.fit(previsores,classes)
        best[j] = [gs.best_estimator_,gs.best_score_]
        
    #resultado do teste com GridSearch
    print("TESTANDO "+str(i) + " MELHORES ATRIBUTOS")
    
    for j in best:
        #mostrando hiperparâmetros dos classificadores
        print(best[j])
    for j in classifiers:
      print("-----------------------" + j + "---------------------")
      cf = best[j][0]
      acuracias = []
      revocs = []
      precisao = []
      fb_scores =[]
      especificidade=[]
      auc=[]
      #realizando novos testes com k_fold de 5 splits para verificar outras métricas
      for train_index, test_index in kf.split(previsores):
          X_train, X_test = previsores[train_index], previsores[test_index]
          y_train, y_test = classes[train_index], classes[test_index]

          y_train = np.reshape(y_train, (y_train.shape[0],))
          cf.fit(X_train, y_train)
          y_pred = cf.predict(X_test)
          
          TN, FP, FN, TP = metrics.confusion_matrix(y_test, y_pred).ravel()
          acuracias.append(metrics.accuracy_score(y_test,y_pred))
          revocs.append(metrics.recall_score(y_test,y_pred))
          fb_scores.append(metrics.fbeta_score(y_test,y_pred,beta = 1.5))
          
          precisao.append(metrics.precision_score(y_test,y_pred))
          #Especificidade — TN / (TN + FP)
          especificidade.append(TN / (TN + FP))
          auc.append(metrics.roc_auc_score(y_test, y_pred))
    
      print("média das acuracias: " + str(np.mean(acuracias)))
      print("média das revocações: " + str(np.mean(revocs)))
      print("média das precisões: " + str(np.mean(precisao)))
      print("média das especificidades: " + str(np.mean(especificidade)))
      print("média das f_betas: " + str(np.mean(fb_scores)))
      print("média das AUCs: " + str(np.mean(auc)))
      
      #guarda o melhor classifcador e o melhor conjunto de atributos, juntamente com suas métricas
      #Importante: Verifica se especificidade não é zero, para evitar selecionar o caso em que um[...]
      #[...] classificador atribui todos os diagnósticos como positivos
      if(np.mean(fb_scores) > melhor_score and np.mean(especificidade) > 0.0):
          melhor_cf_por_k = cf
          melhor_conj_atributos = atts
          melhores_metricas = {'acuracias':np.mean(acuracias), 'revocacos':np.mean(revocs),
                               'precisoes' : np.mean(precisao), 'especificidades':np.mean(especificidade),
                               'f_betas': np.mean(fb_scores), 'AUCs': np.mean(auc)}
          melhor_score = np.mean(fb_scores)

#exibe o resultado final das excuções com selectKBest os atributos selecionados no melhor K, [...]
#[...] Assim como o melhor classificador e se desempenho            
print("MELHOR CONJUNTO DE ATRIBUTOS: " + str(melhor_conj_atributos))


print("MELHOR CLASSIFICADOR")
print(melhor_cf_por_k)
print("média das acuracias: " + str(melhores_metricas['acuracias']))
print("média das revocações: " + str(melhores_metricas['revocacos']))
print("média das precisões: " + str(melhores_metricas['precisoes']))
print("média das especificidades: " + str(melhores_metricas['especificidades']))
print("média das f_betas: " + str(melhores_metricas['f_betas']))
print("média das AUCs: " + str(melhores_metricas['AUCs']))
