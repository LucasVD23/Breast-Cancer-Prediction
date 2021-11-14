# Trabalho 1 de AM1 

## Integrantes ##

- Lucas Vinícius Domingues 769699
- Rafael Yoshio Yamawaki Murata 769681
- Victor Luís Aguilar Antunes 769734

Link para o Dataset utilizado: [Breast Cancer Coimbra Data Set](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Coimbra)

## Organização do Repositório ##

* Arquivo para o melhor classificador obtido nos testes utilizando todos os atributos:
``` AdaBoost_melhor_todos_atributos.py ```

* Arquivo para o melhor classificador obtido nos testes utilizando o *selectKBest*: 
```SVM_melhor_KBest.py```

* Arquivo para o melhor classificador obtido nos testes utilizando os atributos usados no artigo mencionando [[1]](https://bmccancer.biomedcentral.com/articles/10.1186/s12885-017-3877-1):
```Melhor_classificador.py```
(Esse arquivo também é o *Melhor classificador obtido para todo o trabalho*)

* Arquivo para o melhor classificador obtido utilizando os 4 atributos com maior correlação com o diagnóstico: 
```KNN_atributos_maior_correlacao.py```

* Arquivos para 4 estapas de testes descritas no artigo e mencionadas acima:
    - Teste com todos os classifcadores, atributos do artigo e 4 atributos com maior relação: [colab](https://colab.research.google.com/drive/1PEY1pvQlbZrHScX24N77GFTGHzdDFKbC#scrollTo=knsQYtUBs34q) (tempo de execução de 15 minutos para cada teste aproximadamente)
    - Teste pela procura do melhor classificador e melhor conjunto de atributos utilizando *SelectKBest*: ```testes_selectKBest.py``` (tempo de execução de 1h15 min aproximadamente)

(Atenção, teste utilizando *SelectKBest* foi deixado avulso pelo seu tamanho e tempo de execução elevados).
