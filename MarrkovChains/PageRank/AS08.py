# this file uses Markov Chains and different Page Rank algorithms like Matrix Power, StatePropogation, Random walk, and Eigen Analysis.
import numpy as np
from scipy.spatial import distance_matrix
from scipy import linalg as LA
import matplotlib.pyplot as plt


def MatrixPower(t, q):
    
    M = np.loadtxt('M.csv', delimiter=',')
    ConstM = M
    for x in range(t):
        M = M @ ConstM
    qStar = M @ np.transpose(q)
    print(qStar)

def StateProgation(t, q):

    M = np.loadtxt('M.csv', delimiter=',')
    qStar = np.transpose(q)

    for x in range(t):
        qStar = M @ qStar
    print(qStar)

def Randomwalk(q, t_0, t):
    
    M = np.loadtxt('M.csv', delimiter=',')

    newPosition = 0
    for i in range(len(q)):
        if q[i] == 1:
            newPosition = i


    desiredColumn = M[:, newPosition]
    # Do T_0 times
    for i in range(t_0):
        newPosition = np.random.choice(a=range(len(M)), p=desiredColumn)
        desiredColumn = M[:, newPosition]
    
    Count = [0,0,0,0,0,0,0,0,0,0]

    for x in range(t_0, t):
        newPosition = np.random.choice(a=range(len(M)), p=desiredColumn)
        Count[newPosition] += 1
        desiredColumn = M[:, newPosition]

    for x in range(len(Count)):
        Count[x] = Count[x]/(t-t_0)
    
    qstar = Count
    print(qstar)







def EigenAnalysis():
    M = np.loadtxt('M.csv', delimiter=',')
    W, V = np.linalg.eig(M)
    V = V/np.sum(V)
    print(V[:,0].real)



# run each with the following
q = [1,0,0,0,0,0,0,0,0,0]
MatrixPower(2048, q)
StateProgation(2048, q)

Randomwalk(q, 100, 2048)
EigenAnalysis()


print('Question2')
q = [0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]
MatrixPower(28, q)
StateProgation(20, q)


