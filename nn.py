from math import exp
import numpy as np

def sigmoid(x):
    return 1 / (1 + exp(-x))


def summer(W, X):
    return np.dot(W,X)
def error(Y,D):
    E=[]
    for i in range(len(Y)):
        for j in range(len(Y[0])):
            E.append(D[i][j]-Y[i][j])
    return E

W = [[1,0,1],
     [0,1,0],
     [1,1,1]]

X = [[1],[1],[0]]
D=[[1],[0],[0]]
Y=summer(W,X)
print(Y)
E=error(summer(W,X),D)



def simapply(Y):
    for i in range(len(Y)):
        for j in range(len(Y[0])):
            Y[i][j]=sigmoid(Y[i][j])
    return Y
def weightimprove(W,Y,X,D):
    alpha=0.001
    E=error(Y,D)
    epoch=0

    while epoch<400:
        for i in range(len(W)):
            for j in range(len(W[0])):
                W[i][j]=W[i][j]+alpha*E[j]*Y[j][0]


        Y=summer(W,X)
        E=error(Y,D)
        epoch+=1
    print('No of iterations=',epoch)
    return W
#print(simapply(Y))
t=weightimprove(W,Y,X,D)
#print(t)
print('Y is:', summer(t,X))
print('Error is:',error(summer(t,X),D))
print('Weights updated are:',t)

