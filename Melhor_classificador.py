import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.model_selection import KFold

data_cancer = shuffle(pd.read_csv("dataR2.csv"),random_state = 15)

# Para o melhor classificador usa-se somente os atributos Idade, BMI, Glicose e Resistina
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

# o KNN foi o que apresentou melhores metricas
knnc = KNeighborsClassifier   (algorithm='auto', leaf_size=30, metric='minkowski',
		                     metric_params=None, n_jobs=None, n_neighbors=9, p=2,
		                     weights='distance')
                            

fb_scorer = metrics.make_scorer(metrics.fbeta_score,beta=1.5)
kf = KFold(n_splits=5,shuffle = False)

print("-----------------------KNeighborsClassifier---------------------")
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
    knnc.fit(X_train, y_train)
    y_pred = knnc.predict(X_test)

    TN, FP, FN, TP = metrics.confusion_matrix(y_test, y_pred).ravel()

    #Acuracia — (TP + TN) / total
    acuracias.append(metrics.accuracy_score(y_test,y_pred))
    #Revocaçao — TP / (TP + FN)
    revocs.append(metrics.recall_score(y_test,y_pred))
    #Precissao — TP / (TP + FP)
    precisao.append(metrics.precision_score(y_test,y_pred))
    #Especificidade — TN / (TN + FP)
    especificidade.append(TN / (TN + FP))
        
    fb_scores.append(metrics.fbeta_score(y_test,y_pred,beta = 1.5))
    auc.append(metrics.roc_auc_score(y_test, y_pred))

print("CLASSIFICADOR: " + str(knnc))
print("Usando os 4 atributos do artigo: ['Idade', 'BMI', 'Glucose', 'Resistin']")
print("média das acuracias: " + str(np.mean(acuracias)))
print("média das revocações: " + str(np.mean(revocs)))
print("média das precisões: " + str(np.mean(precisao)))
print("média das especificidades: " + str(np.mean(especificidade)))
print("média das f_betas: " + str(np.mean(fb_scores)))
print("média das AUCs: " + str(np.mean(auc)))