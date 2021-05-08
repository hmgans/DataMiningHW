#This is the python file for AS05
from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import pyplot as plt
import numpy
import random
import hashlib
import string






def MisraAlg(textFile):

    labels = {}
    f = open(textFile, 'r')
    filestring = f.read()
    size = len(filestring)
    print(size)

    for i in range(len(filestring)):
        item = filestring[i]
        
        # If item has label increment count 
        if item in labels.keys():
            labels[item] += 1

        # If there is another that can bee added
        elif len(labels.keys()) < 2:
            labels[item] = 1
        else:
            notAdded = True
            for j in labels.keys():
                if labels[j] == 0:
                    labels[item] = labels.pop(j)
                    labels[item] = 1
                    notAdded = False
                    break
            if notAdded:
                for j in labels.keys():
                    labels[j] -= 1
    
    total = 0

    for i in labels.keys():
        print(i) # letter
        print(labels[i]) # count
        print(float(labels[i])/size) # ratio for S1



_memomask = {}

def hash_function(x):
    a = random.randint(10000000,20000000)
    k = random.randint(10000000,20000000)

    def h(var):
        return int(k * (a * ord(var) - int( a * ord(var)))) % 8
    return h


def CountMinSketchAlg(textFile):

    
    # Get 5 different salts for hashing
    salts = [""] * 6
    for s in range(6):
        salts[s] += random.choice(string.ascii_lowercase)

    #initialize table to 0
    table = []
    for i in range(6):
        table.append({})
        for j in range(8):
            table[i][j] = 0

    Set = set()
    f = open(textFile, 'r')
    filestring = f.read()
    size = len(filestring)

    #Input all of the stream
    for i in range(len(filestring)):
        item = filestring[i]
        Set.add(item)

        for j in range(6):
            newString = item + salts[j]
            hashval = hashlib.sha1(newString.encode()).hexdigest()
            index = int(hashval, 16) % 8
            table[j][index] += 1


    # Print values for everything
    S = []
    for i in range(len(Set)):
        S.append(Set.pop())

    SVals = {}
    for i in range(len(S)):
        values = []
        for j in range(6):
            newString = S[i] + salts[j]
            hashval = hashlib.sha1(newString.encode()).hexdigest()
            index = int(hashval, 16) % 8
            values.append(table[j][index])
        values.sort()
        SVals[S[i]] = values[0]


    sorted_keys = sorted(SVals, key=SVals.get, reverse=True)
    current = 0
    for w in sorted_keys:
        if current == 7:
            break
        print(w)
        print(SVals[w])
        print(float(SVals[w])/size)
        current += 1




    




    
    
    
            





#CountMinSketchAlg("S2.txt")
#CountMinSketchAlg("S3.txt")



#debugging Purpose
#MisraAlg("AS5/S1.txt")

MisraAlg("S3.txt")