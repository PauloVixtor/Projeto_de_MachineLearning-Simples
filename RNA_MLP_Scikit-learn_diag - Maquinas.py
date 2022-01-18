"""
Created on Nov 13 2021 - @author: Paulo
"""
import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
#from sklearn.model_selection import train_test_split

#cria uma matriz e armazena nela o conjunto de treinamento
ct = np.array([[ 1,	1,	0,	1,	1],
               [ 0,	0,	1,	0,	0],
               [ 1,	1,	0,	0,	0],
               [ 1,	0,	1,	1,	1],
               [ 1,	0,	0,	1,	0],
               [ 0,	0,	1,	1,	1]])

#Separa as colunas do conjunto de treinamento em sintomas (s) e diagnóstico (d) 
p = ct[:,0:4]
d = ct[:,4]
X_train = p
y_train = d
X_test = p
y_test = d

#Se quiser usar o perceptron
#clf = Perceptron(tol=1e-5, random_state=0)

#Se quiser usar MLP
#Cria a RNA do tipo Multi-layer Perceptron com a configuração indicada
clf = MLPClassifier(hidden_layer_sizes=(10,), learning_rate='constant', 
                    learning_rate_init=0.005, max_iter=3000)

#Treina a RNA
clf.fit(X_train, y_train)

#Calcula e imprime a acurácia do treinamento
acuracia_trein = clf.score(X_test, y_test)
print("Acurácia do Treinamento = %.2f%%" %(acuracia_trein*100.0))

#cria uma matriz com os sintomas dos novos casos (Luis, Laura, Lucia)
Novos_Registros = np.array([[0, 0, 0, 1],
                            [1, 1, 1, 1],
                            [1, 1, 0, 0],
                            [1, 1, 0, 1],
                            [1, 1, 1, 0]])

#Faz a predição para os novos casos e armazena no vetor de respostas (resp)
resp = clf.predict(Novos_Registros)

#Exibe os resultados da RNA MLP para os novos casos 
print("\nPredição da RNA para R1")
if resp[0] > 0.5: print("FALHA")
else: print("NORMAL")
    
print("\nPredição da RNA para R2")
if resp[1] > 0.5: print("FALHA")
else: print("NORMAL")

print("\nPredição da RNA para R3")
if resp[2] > 0.5: print("FALHA")
else: print("NORMAL")

print("\nPredição da RNA para R4")
if resp[3] > 0.5: print("FALHA")
else: print("NORMAL")

print("\nPredição da RNA para R5(MICHEL)")
if resp[4] > 0.5: print("FALHA")
else: print("NORMAL")

