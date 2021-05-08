#This is the python file for AS01
from random import random
from random import randint
import numpy as np
import matplotlib.pyplot as plt
import time
import random

def create2GramWord(filelocation):
    #open file
    f = open(filelocation, "r")

    # store file as string
    filestring = f.read()
    words = filestring.split(' ')
    #close file
    f.close()

    dictionary = dict()
    # iterate through all but last since it is a 2-Gram
    for i in range(len(words) - 1):
        gram = words[i] + " " + words[i+1]

        dictionary[gram] = gram
    return dictionary

def create3GramWord(filelocation):
    #open file
    f = open(filelocation, "r")

    # store file as string
    filestring = f.read()
    words = filestring.split(' ')
    #close file
    f.close()

    dictionary = dict()
    # iterate through all but last since it is a 2-Gram
    for i in range(len(words) - 2):
        gram = words[i] + " " + words[i+1] + " " + words[i+2]
        dictionary[gram] = gram
    return dictionary

def create3GramChar(filelocation):
    #open file
    f = open(filelocation, "r")

    # store file as string
    filestring = f.read()
    #close file
    f.close()

    dictionary = dict()
    # iterate through all but last since it is a 3-Gram
    for i in range(len(filestring) - 2):
        gram = filestring[i] + filestring[i+1] + filestring[i+2]
        dictionary[gram] = gram
    return dictionary

def computeJaccardSimilarity(set1, set2):
    info1 = set1.keys()
    info2 = set2.keys()
    numberOfSimilar = 0

    size1 = len(info1)
    size2 = len(info2)

    for key in set1:
        if(key in set2):
            numberOfSimilar += 1
    total = size1 + size2
    # similar divided by total with no repeats
    return float(numberOfSimilar)/(total-numberOfSimilar)

def minHashFunction(t, s1, s2):

    info1 = s1.keys()
    info2 = s2.keys()

    dictionary = dict()

    for i in info1:
        dictionary[i] = i
    for i in info2:
        dictionary[i] = i

    vector1 = dictionary.copy()
    vector2 = dictionary.copy()

    # create vectors
    for i in dictionary:
        if(i in info1):
            vector1[i] = 1
        else:
            vector1[i] = 0
        if(i in info2):
            vector2[i] = 1
        else:
            vector2[i] = 0

    listOfKGrams = []
    for i in dictionary.keys():
        listOfKGrams.append(i)
    #perform calculations
    totalSim = 0
    scalar = float(1)/t
    for i in range(t):
        random.shuffle(listOfKGrams)
        first1 = None
        first2 = None
        for j in range(len(listOfKGrams)):
            gram = listOfKGrams[j]

            if vector1[gram] == 1 and first1 is None:
                first1 = gram
            if(vector2[gram] == 1 and first2 is None):
                first2 = gram

            if(first1 is not None and first2 is not None):
                break

        if(first1 == first2):
            totalSim += float(scalar)
        first1 = None
        first2 = None
    return totalSim

def minHashFunction2(t, s1, s2):

    info1 = s1.keys()
    info2 = s2.keys()


    listOfValues = list(set().union(info1, info2))

    scalar = float(1)/t
    bounds = len(listOfValues)-1

    totalSim = float(0)
    for i in range(t):
        randomInt = randint(0, bounds)
        value = listOfValues[randomInt]
        if(value in s1 and value in s2):
            totalSim += scalar
    return totalSim




def timingCollision(s1, s2):

    arrayForTime100 = []
    arrayForResult100 = []
    for i in range(100):

        startTime = time.time()
        result = minHashFunction(100, s1, s2)
        stopTime = time.time()

        # compute average time of timesToLoop
        averageTime = stopTime - startTime
        arrayForTime100.append(averageTime)
        arrayForResult100.append(result)

    arrayForTime200 = []
    arrayForResult200 = []

    for i in range(100):

        startTime = time.time()
        result = minHashFunction(200, s1, s2)
        stopTime = time.time()

        # compute average time of timesToLoop
        averageTime = stopTime - startTime
        arrayForTime200.append(averageTime)
        arrayForResult200.append(result)

    arrayForTime400 = []
    arrayForResult400 = []
    for i in range(100):

        startTime = time.time()
        result = minHashFunction(400, s1, s2)
        stopTime = time.time()

        # compute average time of timesToLoop
        averageTime = stopTime - startTime
        arrayForTime400.append(averageTime)
        arrayForResult400.append(result)

    arrayForTime800 = []
    arrayForResult800 = []
    for i in range(100):

        startTime = time.time()
        result = minHashFunction(800, s1, s2)
        stopTime = time.time()

        # compute average time of timesToLoop
        averageTime = stopTime - startTime
        arrayForTime800.append(averageTime)
        arrayForResult800.append(result)

    arrayForTime1600 = []
    arrayForResult1600 = []
    for i in range(100):

        startTime = time.time()
        result = minHashFunction(1600, s1, s2)
        stopTime = time.time()

        # compute average time of timesToLoop
        averageTime = stopTime - startTime
        arrayForTime1600.append(averageTime)
        arrayForResult1600.append(result)

    t100 = (arrayForTime100, arrayForResult100)
    t200 = (arrayForTime200, arrayForResult200)
    t400 = (arrayForTime400, arrayForResult400)
    t800 = (arrayForTime800, arrayForResult800)
    t1600 = (arrayForTime1600, arrayForResult1600)
    data = (t100, t200, t400, t800, t1600)
    colors = ("orange", "red", "green", "blue", "yellow")
    groups = ("t=100", "t=200", "t=400", "t=800", "t=1600")

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)

    for data, color, group in zip(data, colors, groups):
        x, y = data
        ax.scatter(x, y, alpha=0.8, c=color, edgecolors='none', s=30, label=group)

    plt.title('Times and Similarity for Min Hash')
    plt.xlabel("Time Seconds")
    plt.ylabel("Similarity")
    plt.legend(loc=1)
    plt.show()









gramD1 = create3GramChar("D1.txt")
gramD2 = create3GramChar("D2.txt")
gramD3 = create3GramChar("D3.txt")
gramD4 = create3GramChar("D4.txt")
#print(computeJaccardSimilarity(gramD1, gramD2))
#print(computeJaccardSimilarity(gramD1, gramD3))
#print(computeJaccardSimilarity(gramD1, gramD4))
#print(computeJaccardSimilarity(gramD3, gramD2))
#print(computeJaccardSimilarity(gramD4, gramD2))
#print(computeJaccardSimilarity(gramD4, gramD3))
print(minHashFunction(1600, gramD1, gramD2))
timingCollision(gramD1, gramD2)
