#Implementación modelo regresión lineal 
#Juan Daniel Aranda Morales - A01379571

#Importando librerías 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Importando dataset
Xtrain = pd.read_csv("train_X.csv")
Ytrain = pd.read_csv("train_Y.csv")

Xtest = pd.read_csv("test_X.csv")
Ytest = pd.read_csv("test_Y.csv")

#Función sigmoide
def sigmoid(x):
    return 1/(1+np.exp(-x))

#Función de costo
def cost(m,A,Y):
    return -(1/m) * np.sum(Y*np.log(A) + (1-Y) * np.log(1-A))

#Función principal
def modelLinealRegresion(X, Y, learningRate, iterations):
    m = X.shape[1]
    n = X.shape[0]

    w = np.zeros((n,1))
    b = 0

    cost_list = []

    for i in range(iterations):
        Z = np.dot(w.T,X) + b
        
        A = sigmoid(Z)
        

        #Función de costo
        cost_op = cost(m,A,Y)

        #Descenso del gradiente
        dW = (1/m)*np.dot(A-Y, X.T)
        dB = (1/m)*np.sum(A - Y)

        w = w - learningRate*dW.T
        b = b - learningRate*dB

        #Obtención de los costos de cada iteración
        cost_list.append(cost_op)

        if(i%(iterations/10)==0):
            print("Costo después de la iteración",i,"Es: ",cost_op)

    return w,b,cost_list


#Procesando dataset

#Eliminar columna ID
Xtrain = Xtrain.drop("Id", axis = 1)
Ytrain = Ytrain.drop("Id", axis = 1)
Xtest = Xtest.drop("Id", axis = 1)
Ytest = Ytest.drop("Id", axis = 1)

#Obtención de los valores numéricos del dataset
Xtrain = Xtrain.values
Ytrain = Ytrain.values
Xtest = Xtest.values
Ytest = Ytest.values

#Formateando dataset
Xtrain = Xtrain.T
Ytrain = Ytrain.reshape(1, Xtrain.shape[1])

Xtest = Xtest.T
Ytest = Ytest.reshape(1, Xtest.shape[1])

learning_rate = 0.0015
iterations = 100000

w, b, cost_list = modelLinealRegresion(Xtrain, Ytrain, learning_rate, iterations)

plt.plot(np.arange(iterations), cost_list)
plt.show()


#Función que calcula la exactitud
def accuracy(X, Y, W, B):
    Z = np.dot(W.T, X) + B
    A = sigmoid(Z)
    
    A = A > 0.5
    
    A = np.array(A, dtype = 'int64')
    
    acc = (1 - np.sum(np.absolute(A - Y))/Y.shape[1])*100
    
    print("Exactitud del modelo es : ", round(acc, 2), "%")

accuracy(Xtest, Ytest, w, b)











