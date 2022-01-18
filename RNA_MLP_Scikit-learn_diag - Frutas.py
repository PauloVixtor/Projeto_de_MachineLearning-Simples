"""
Created on Nov 13 2021 - @author: Paulo
"""
import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
#from sklearn.model_selection import train_test_split

#cria uma matriz e armazena nela o conjunto de treinamento
ct = np.array([[0.1, 0.4, 0.7, 1],
               [0.5, 0.7, 0.1, 1],
               [0.6, 0.9, 0.8, 0],
               [0.3, 0.7, 0.2, 0]])

#Separa as colunas do conjunto de treinamento em sintomas (s) e diagnóstico (d) 
p = ct[:, 0:3]
d = ct[:, 3]
X_train = p
y_train = d
X_test = p
y_test = d

#Se quiser usar o perceptron
#clf = Perceptron(tol=1e-5, random_state=0)

#Se quiser usar MLP
#Cria a RNA do tipo Multi-layer Perceptron com a configuração indicada
clf = MLPClassifier(hidden_layer_sizes=(10,), learning_rate='constant',
                    learning_rate_init=0.001, max_iter=3000)

#Treina a RNA
clf.fit(X_train, y_train)

#Calcula e imprime a acurácia do treinamento
acuracia_trein = clf.score(X_test, y_test)
print("Acurácia do Treinamento = %.2f%%" %(acuracia_trein*100.0))

#cria uma matriz com os sintomas dos novos casos (Luis, Laura, Lucia)
Novos_Registros = np.array([[0.1, 0.4, 0.1],
                            [0.6, 0.9, 0.2],
                            [0.5, 0.7, 0.1],
                            [0.3, 0.9, 0.1],
                            [0.3, 0.1, 0.5]])

#Faz a predição para os novos casos e armazena no vetor de respostas (resp)
resp = clf.predict(Novos_Registros)
print(f'\nA resposta foi : {resp}')

#Exibe os resultados da RNA MLP para os novos casos 
print("\nPredição da RNA para Caso 1")
if resp[0] > 0.5: print("TANGERINA")
else: print("LARANJA")
    
print("\nPredição da RNA para Caso 2")
if resp[1] > 0.5: print("TANGERINA")
else: print("LARANJA")

print("\nPredição da RNA para Caso 3")
if resp[2] > 0.5: print("TANGERINA")
else: print("LARANJA")

print("\nPredição da RNA para Caso 4")
if resp[3] > 0.5: print("TANGERINA")
else: print("LARANJA")

print("\nPredição da RNA para Caso 5 (MICHEL)")
if resp[4] > 0.5: print("TANGERINA")
else: print("LARANJA")