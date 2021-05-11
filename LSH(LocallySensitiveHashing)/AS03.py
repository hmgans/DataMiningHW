#This is the python file for AS03
# LSH is the process of using multiple hash functions to compute simularity.
# Compute simularity using f(s) = 1 - (1-S^{b})^{r} where S is the simularity of two sets, b = # of bands, r = # of hash functions

from numpy import *
from random import random
from random import randint
import numpy as np
import matplotlib.pyplot as plt
import time
import random
import math
import csv

def varyingRandB():
    t=linspace(0, 1, 100)
    a = 1 - (1-t**20)**8
    b = 1 - (1-t**16)**10
    c = 1 - (1-t**10)**16
    d = 1 - (1-t**8)**20
    fig, ax = plt.subplots()

    plt.plot(t, a, 'b', label='r=8, b=20') # plotting t, a separately 
    plt.plot(t, b, 'r', label='r=10, b=16') # plotting t, b separately 
    plt.plot(t, c, 'y', label='r=16, b=10') # plotting t, c separately 
    plt.plot(t, d, 'g', label='r=20, b=8') # plotting t, c separately
    plt.xlabel("Tau")
    plt.ylabel("Probabilty")
    plt.legend()
    plt.show()

# Cretae unit vecotrs to  then compare dot products latere
def createUnitVectors(dim, t):

    vectors = []
    for i in range(t):
        list = []
        for j in range(dim):
            u = random.uniform(0.0, 1.0)
            unot = random.uniform(0.0, 1.0)
            g = np.sqrt(-2*np.log(u))*np.cos(2*np.pi * unot)
            list.append(g)
        #At this point U is generated so we need to normalize it
        vecG = np.array(list)
        normG = np.linalg.norm(vecG)
        unitG = vecG/normG
        vectors.append(unitG)
    return vectors

def calculateDotProducts(vectors):
    products = []
    for i in range(len(vectors)):
        for j in range(i+1,len(vectors)):
            dotProduct = np.dot(vectors[i], vectors[j])
            products.append(dotProduct)
    return products

# Plot Dot product 
def plotCFGForDotProducts(dotP):

    m = len(dotP)

    values, base = np.histogram(dotP, bins=m);
    #evaluate the cumulative
    cumulative = np.cumsum(values);

    # plot the cumulative function
    X2 = np.sort(dotP);

    F2 = np.array(range(m))/float(m);

    #Label Graph
    plt.title('Cumulative Distribution of Dot Products where t=200 and d=120');
    plt.xlabel('Percent Of Parings');
    plt.ylabel('Added Parings');

    plt.plot(X2, F2)
    plt.xlim([-1,1])

    plt.show()



#Part 3

def normalizedDataPoints():
    vectors = []
    with open('R.csv') as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            u = []
            for i in range(len(row)):
                
                u.append(float(row[i]))
            
            vec = np.array(u)
            norm = np.linalg.norm(vec)
            unit = vec/norm
            vectors.append(unit)
    return vectors
               
           
# Find Angular simularaties above a certain threshold Tau
def angSim(a,b):
    result = 1 - ((1/np.pi)*np.arccos(np.dot(a,b)))
    return result

def AngSims(vectors):
    Angsims = []
    for i in range(len(vectors)):
        for j in range(i+1,len(vectors)):
            angSimResult = angSim(vectors[i], vectors[j])
            Angsims.append(angSimResult)
    return Angsims

def numberAboveTau(Simalarities, threshold):
    total = 0
    for i in range(len(Simalarities)):
        if(Simalarities[i]>threshold):
            total += 1
    return total;


def plotCFGForSimilaritiesA(Sims):

    m = len(Sims)

    values, base = np.histogram(Sims, bins=m);
    #evaluate the cumulative
    cumulative = np.cumsum(values);

    # plot the cumulative function
    X2 = np.sort(Sims);

    F2 = np.array(range(m))/float(m);

    #Label Graph
    plt.title('Cumulative Distribution of Angular Similarties here n=450 and d=100');
    plt.xlabel('Percent Of Parings');
    plt.ylabel('Added Parings');

    plt.plot(X2, F2)
    plt.xlim([0,1])

    plt.show()

def plotCFGForSimilaritiesB(Sims):

    m = len(Sims)

    values, base = np.histogram(Sims, bins=m);
    #evaluate the cumulative
    cumulative = np.cumsum(values);

    # plot the cumulative function
    X2 = np.sort(Sims);

    F2 = np.array(range(m))/float(m);

    #Label Graph
    plt.title('Cumulative Distribution of Angular Similarties here t=200 and d=120');
    plt.xlabel('Percent Of Parings');
    plt.ylabel('Added Parings');

    plt.plot(X2, F2)
    plt.xlim([0,1])

    plt.show()

varyingRandB()

VectorsFor2 = createUnitVectors(120,200)
DotProducts2 = calculateDotProducts(VectorsFor2)
plotCFGForDotProducts(DotProducts2)

#getVectorsfromCSV and normalize them

VectorsFor3A = normalizedDataPoints()
AngSims3A = AngSims(VectorsFor3A)
print(numberAboveTau(AngSims3A, 0.8))
plotCFGForSimilaritiesA(AngSims3A)

VectorsFor3B = createUnitVectors(120,200)
AngSims3B = AngSims(VectorsFor3B)
print(numberAboveTau(AngSims3B, 0.75))
plotCFGForSimilaritiesB(AngSims3B)
normalizedDataPoints()