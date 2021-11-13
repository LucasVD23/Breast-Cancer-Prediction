import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn import svm
from sklearn import metrics
from sklearn.model_selection import KFold

data_cancer = shuffle(pd.read_csv("dataR2.csv"),random_state = 15)

# Usando somente os atributos Idade, BMI, Glicose e Resistina
data_cancer = data_cancer.drop(columns=['Insulin','HOMA','Leptin','Adiponectin', 'MCP.1'])
previsores = np.array(data_cancer.iloc[:,0:4].values)
classes = np.array(data_cancer.iloc[:,4:5].values)
scaler = MinMaxScaler(feature_range=(0,1))
previsores = scaler.fit_transform(previsores)
classes = np.reshape(classes, (classes.shape[0],))
for i in range(0,len(classes)):
    if(classes[i] == 1 ):
        classes[i] = 0
    else:
        classes[i] = 1

# Hiperparametros obtidos pelo artigo “Using Resistin, glucose, age and BMI to predict the presence of breast cancer”
dtc = DecisionTreeClassifier  (ccp_alpha=0.0, class_weight=None, criterion='gini',
		                       max_depth=14, max_features=None, max_leaf_nodes=None,
		                       min_impurity_decrease=0.0, min_impurity_split=None,
		                       min_samples_leaf=4, min_samples_split=2,
		                       min_weight_fraction_leaf=0.0, presort='deprecated',
		                       random_state=None, splitter='best')
                            
rfc = RandomForestClassifier  (bootstrap=True, ccp_alpha=0.0, class_weight=None,
		                       criterion='entropy', max_depth=16, max_features='auto',
		                       max_leaf_nodes=None, max_samples=None,
		                       min_impurity_decrease=0.0, min_impurity_split=None,
		                       min_samples_leaf=4, min_samples_split=2,
		                       min_weight_fraction_leaf=0.0, n_estimators=100,
		                       n_jobs=None, oob_score=False, random_state=None,
		                       verbose=0, warm_start=False)
                            
knnc = KNeighborsClassifier   (algorithm='auto', leaf_size=30, metric='minkowski',
		                     metric_params=None, n_jobs=None, n_neighbors=9, p=2,
		                     weights='distance')
                            
svmc = svm.SVC                (C=1e-06, break_ties=False, cache_size=200, class_weight=None, coef0=0.0,
		    decision_function_shape='ovr', degree=3, gamma='scale', kernel='linear',
		    max_iter=-1, probability=False, random_state=None, shrinking=True,
		    tol=0.001, verbose=False)
                            
adac = AdaBoostClassifier     (algorithm='SAMME.R', base_estimator=None, learning_rate=0.4,
		                   n_estimators=11, random_state=None)

classifiers = {'Decision Tree': dtc, 'RandomForest': rfc, 'KNN': knnc, 'SVM': svmc, 'AdaBoost': adac}

fb_scorer = metrics.make_scorer(metrics.fbeta_score,beta=1.5)
kf = KFold(n_splits=5,shuffle = False)

melhor_cf_por_k = dtc
melhor_score = 0
melhor_conj_atributos = []
melhores_metricas = {}
conjuntos_atts =[]

for i in classifiers:
    print("-----------------------" + i + "---------------------")
    cf = classifiers[i]
    acuracias = []
    revocs = []
    fb_scores =[]
    precisao=[]
    especificidade=[]
    auc=[]

    for train_index, test_index in kf.split(previsores):
        X_train, X_test = previsores[train_index], previsores[test_index]
        y_train, y_test = classes[train_index], classes[test_index]
        y_train = np.reshape(y_train, (y_train.shape[0],))
        cf.fit(X_train, y_train)
        y_pred = cf.predict(X_test)

        TN, FP, FN, TP = metrics.confusion_matrix(y_test, y_pred).ravel()

        #Acuracia — (TP + TN) / total
        acuracias.append(metrics.accuracy_score(y_test,y_pred))
        #Revocaçao — TP / (TP + FN)
        revocs.append(metrics.recall_score(y_test,y_pred))
        #Precissao — TP / (TP + FP)
        precisao.append(metrics.precision_score(y_test,y_pred))
        #Especificidade — TN / (TN + FP)
        especificidade.append(TN / (TN + FP))
        
        fb_scores.append(metrics.fbeta_score(y_test,y_pred,1.5))
        auc.append(metrics.roc_auc_score(y_test, y_pred))

    print("média das acuracias: " + str(np.mean(acuracias)))
    print("média das revocações: " + str(np.mean(revocs)))
    print("média das precisões: " + str(np.mean(precisao)))
    print("média das especificidades: " + str(np.mean(especificidade)))
    print("média das f_betas: " + str(np.mean(fb_scores)))
    print("média das AUCs: " + str(np.mean(auc)))

    if(np.mean(fb_scores) > melhor_score and np.mean(especificidade) > 0.0):
          melhor_cf_por_k = cf
          melhores_metricas = {'acuracias':np.mean(acuracias), 'revocacos':np.mean(revocs),
                               'precisoes' : np.mean(precisao), 'especificidades':np.mean(especificidade),
                               'f_betas': np.mean(fb_scores), 'AUCs': np.mean(auc)}
          melhor_score = np.mean(fb_scores)

print("MELHOR CLASSIFICADOR")
print(melhor_cf_por_k)
print("média das acuracias: " + str(melhores_metricas['acuracias']))
print("média das revocações: " + str(melhores_metricas['revocacos']))
print("média das precisões: " + str(melhores_metricas['precisoes']))
print("média das especificidades: " + str(melhores_metricas['especificidades']))
print("média das f_betas: " + str(melhores_metricas['f_betas']))
print("média das AUCs: " + str(melhores_metricas['AUCs']))
