#This is the python file for AS04
# This file implements the use of the three hierarchical clustering methods: single link, complete link, and mean link for clustering
# It also runs three Assignment Based clustering methods: K++Means, Gonzalez, and Lloyds Algorithm

from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import pyplot as plt
import numpy
import random

def textFileReadingAndStorage(name):
    f = open(name, 'r')
    content = f.readlines()
    temp = content[0]
    constructor = temp.split()
    dim = len(constructor)
    points = len(content)
    arr = [[0 for i in range(dim)] for j in range(points)]


    for i in range(points):
        nums = content[i].split()
        for j in range(dim):
            arr[i][j] = float(nums[j])

    return arr

# Single Link,  find closest point and adds it to the cluster
def singleLinkClusters(arr):
    result = linkage(arr, 'single')
    fig = plt.figure(figsize=(18,7))
    plt.title("Groupings for Single Link on C1")
    plt.xlabel("Point Number")
    dn = dendrogram(result)
    plt.show()

    group1x = [arr[19][0], arr[0][0], arr[1][0]]
    group1y = [arr[19][1], arr[0][1], arr[1][1]]
    group2x = [arr[17][0], arr[18][0], arr[13][0], arr[11][0], arr[10][0], arr[12][0]]
    group2y = [arr[17][1], arr[18][1], arr[13][1], arr[11][1], arr[10][1], arr[12][1]]
    group3x = [arr[16][0]]
    group3y = [arr[16][1]]
    group4x = [arr[14][0], arr[15][0]]
    group4y = [arr[14][1], arr[15][1]]
    group5x = [arr[8][0], arr[7][0], arr[8][0], arr[2][0], arr[3][0], arr[4][0], arr[5][0], arr[6][0]]
    group5y = [arr[8][1], arr[7][1], arr[8][1], arr[2][1], arr[3][1], arr[4][1], arr[5][1], arr[6][1]]


    fig = plt.figure(figsize=(18,7))
    ax1 = fig.add_subplot()

    ax1.scatter(group1x, group1y, c='b', label='Group 1')
    ax1.scatter(group2x, group2y, c='r', label='Group 2')
    ax1.scatter(group3x, group3y, c='y', label='Group 3')
    ax1.scatter(group4x, group4y, c='g', label='Group 4')
    ax1.scatter(group5x, group5y, c='c', label='Group 5')
    plt.legend(loc='upper left')
    plt.title("Groupings for Single Link on C1 when K=5")

    plt.show()

# Complete Link adds the minimum maximum point to the all points in the cluster.
def completeLinkClusters(arr):
    result = linkage(arr, 'complete')
    fig = plt.figure(figsize=(18,7))
    plt.title("Groupings for Complete Link on C1")
    plt.xlabel("Point Number")
    dn = dendrogram(result)
    plt.show()

    group1x = [arr[16][0], arr[14][0], arr[15][0]]
    group1y = [arr[16][1], arr[14][1], arr[15][1]]
    group2x = [arr[13][0], arr[11][0], arr[10][0], arr[12][0], arr[17][0], arr[18][0]]
    group2y = [arr[13][1], arr[11][1], arr[10][1], arr[12][1], arr[17][1], arr[18][1]]
    group3x = [arr[2][0], arr[3][0], arr[4][0]]
    group3y = [arr[2][1], arr[3][1], arr[4][1]]
    group4x = [arr[19][0], arr[0][0], arr[1][0]]
    group4y = [arr[19][1], arr[0][1], arr[1][1]]
    group5x = [arr[5][0], arr[6][0], arr[9][0], arr[7][0], arr[8][0]]
    group5y = [arr[5][1], arr[6][1], arr[9][1], arr[7][1], arr[8][1]]


    fig = plt.figure(figsize=(18,7))
    ax1 = fig.add_subplot()

    ax1.scatter(group1x, group1y, c='b', label='Group 1')
    ax1.scatter(group2x, group2y, c='r', label='Group 2')
    ax1.scatter(group3x, group3y, c='y', label='Group 3')
    ax1.scatter(group4x, group4y, c='g', label='Group 4')
    ax1.scatter(group5x, group5y, c='c', label='Group 5')
    plt.legend(loc='upper left')
    plt.title("Groupings for Complete Link on C1 when K=5")

    plt.show()

# Mean Links adds the minimum average point to the cluster.
def meanLinkClusters(arr):
    result = linkage(arr, 'average')
    fig = plt.figure(figsize=(18,7))
    plt.title("Groupings for Mean Link on C1")
    plt.xlabel("Point Number")
    dn = dendrogram(result)
    plt.show()

    group1x = [arr[16][0]]
    group1y = [arr[16][1]]
    group2x = [arr[13][0], arr[11][0], arr[10][0], arr[12][0], arr[17][0], arr[18][0]]
    group2y = [arr[13][1], arr[11][1], arr[10][1], arr[12][1], arr[17][1], arr[18][1]]
    group3x = [arr[14][0], arr[15][0]]
    group3y = [arr[14][1], arr[15][1]]
    group4x = [arr[19][0], arr[0][0], arr[1][0]]
    group4y = [arr[19][1], arr[0][1], arr[1][1]]
    group5x = [arr[2][0], arr[3][0], arr[4][0], arr[5][0], arr[6][0], arr[9][0], arr[7][0], arr[8][0]]
    group5y = [arr[2][1], arr[3][1], arr[4][1], arr[5][1], arr[6][1], arr[9][1], arr[7][1], arr[8][1]]


    fig = plt.figure(figsize=(18,7))
    ax1 = fig.add_subplot()

    ax1.scatter(group1x, group1y, c='b', label='Group 1')
    ax1.scatter(group2x, group2y, c='r', label='Group 2')
    ax1.scatter(group3x, group3y, c='y', label='Group 3')
    ax1.scatter(group4x, group4y, c='g', label='Group 4')
    ax1.scatter(group5x, group5y, c='c', label='Group 5')
    plt.legend(loc='upper left')
    plt.title("Groupings for Mean Link on C1 when K=5")

    plt.show()

global phiG

def plotScatterForGonz(arr, centers, phiG):

    fig = plt.figure(figsize=(18,7))
    ax1 = fig.add_subplot()
    group1x = []
    group1y = []
    group2x = []
    group2y = []
    group3x = []
    group3y = []
    group4x = []
    group4y = []
    centerX = [centers[0][0], centers[1][0], centers[2][0], centers[3][0]]
    centerY = [centers[0][1], centers[1][1], centers[2][1], centers[3][1]]



    for j in range(len(arr)):
        if phiG[j] == 0:
            group1x.append(arr[j][0])
            group1y.append(arr[j][1])
        if phiG[j] == 1:
            group2x.append(arr[j][0])
            group2y.append(arr[j][1])

        if phiG[j] == 2:
            group3x.append(arr[j][0])
            group3y.append(arr[j][1])
        if phiG[j] == 3:
            group4x.append(arr[j][0])
            group4y.append(arr[j][1])

    ax1.scatter(group1x, group1y, c='b', label='Group 1')
    ax1.scatter(group2x, group2y, c='r', label='Group 2')
    ax1.scatter(group3x, group3y, c='y', label='Group 3')
    ax1.scatter(group4x, group4y, c='g', label='Group 4')
    ax1.scatter(centerX, centerY, c='k', label='Centers')
    plt.legend(loc='upper left')
    plt.title("Centers and Clusters for Gonzalez")

    plt.show()

def plotScatterForKMean(arr, centers):

    fig = plt.figure(figsize=(18,7))
    ax1 = fig.add_subplot()
    group1x = []
    group1y = []
    group2x = []
    group2y = []
    group3x = []
    group3y = []
    group4x = []
    group4y = []
    centerX = [centers[0][0], centers[1][0], centers[2][0], centers[3][0]]
    centerY = [centers[0][1], centers[1][1], centers[2][1], centers[3][1]]



    for j in range(len(arr)):
        a = numpy.array(centers[0])
        b = numpy.array(arr[j])
        loc = 0
        distance = numpy.linalg.norm(a-b)
        for k in range(1, len(centers)):
            a = numpy.array(centers[k])
            tempdistance = numpy.linalg.norm(a-b)
            if tempdistance < distance:
                distance = tempdistance
                loc = k
        if loc == 0:
            group1x.append(arr[j][0])
            group1y.append(arr[j][1])
        if loc == 1:
            group2x.append(arr[j][0])
            group2y.append(arr[j][1])

        if loc == 2:
            group3x.append(arr[j][0])
            group3y.append(arr[j][1])
        if loc == 3:
            group4x.append(arr[j][0])
            group4y.append(arr[j][1])


    ax1.scatter(group1x, group1y, c='b', label='Group 1')
    ax1.scatter(group2x, group2y, c='r', label='Group 2')
    ax1.scatter(group3x, group3y, c='y', label='Group 3')
    ax1.scatter(group4x, group4y, c='g', label='Group 4')
    ax1.scatter(centerX, centerY, c='k', label='Centers')
    plt.legend(loc='upper left')
    plt.title("Centers and Clusters for KMean")

    plt.show()

def plotScatterForLloyd(arr, centers):

    fig = plt.figure(figsize=(18,7))
    ax1 = fig.add_subplot()
    group1x = []
    group1y = []
    group2x = []
    group2y = []
    group3x = []
    group3y = []
    group4x = []
    group4y = []
    centerX = [centers[0][0], centers[1][0], centers[2][0], centers[3][0]]
    centerY = [centers[0][1], centers[1][1], centers[2][1], centers[3][1]]



    for j in range(len(arr)):
        a = numpy.array(centers[0])
        b = numpy.array(arr[j])
        loc = 0
        distance = numpy.linalg.norm(a-b)
        for k in range(1, len(centers)):
            a = numpy.array(centers[k])
            tempdistance = numpy.linalg.norm(a-b)
            if tempdistance < distance:
                distance = tempdistance
                loc = k
        if loc == 0:
            group1x.append(arr[j][0])
            group1y.append(arr[j][1])
        if loc == 1:
            group2x.append(arr[j][0])
            group2y.append(arr[j][1])

        if loc == 2:
            group3x.append(arr[j][0])
            group3y.append(arr[j][1])
        if loc == 3:
            group4x.append(arr[j][0])
            group4y.append(arr[j][1])


    ax1.scatter(group1x, group1y, c='b', label='Group 1')
    ax1.scatter(group2x, group2y, c='r', label='Group 2')
    ax1.scatter(group3x, group3y, c='y', label='Group 3')
    ax1.scatter(group4x, group4y, c='g', label='Group 4')
    ax1.scatter(centerX, centerY, c='k', label='Centers')
    plt.legend(loc='upper left')
    plt.title("Centers and Clusters for Lloyd when Centers are the First four points")

    plt.show()

# Gonzalez's Algorthm will select a center at random and then will chose the next center as the farthest point from that center.
def GonzalezAlgorthim(arr, k):
    n = len(arr)

    #initialize s1 or first center
    s = [[0 for i in range(len(arr[0]))] for j in range(k)]
    s[0] = arr[0] # s1=x1 center is first point
    global phiG

    phiG = [0 for i in range(n)] #Assign All points to first center point 

    for i in range(1,k):
        gdistance = 0.0
        s[i] = arr[0] #si = x1
        for j in range(n):
            xj = arr[j]
            center = s[phiG[j]]
            a = numpy.array(xj)
            b = numpy.array(center)
            pdistance = numpy.linalg.norm(a-b)

            if(pdistance > gdistance):
                gdistance = pdistance
                s[i] = xj
        for j in range(n): #update center for all points
            xj = arr[j]
            center = s[phiG[j]]
            si = s[i]
            x = numpy.array(xj)
            sphi = numpy.array(center)
            stemp = numpy.array(si)
            firstDistance = numpy.linalg.norm(x-sphi)
            secondDistance = numpy.linalg.norm(x-stemp)
            if(firstDistance>secondDistance):
                phiG[j] = i
    
    return s

# K Means++ finds the starting center points by selecting the next center based on distance
def KMeansPP(arr, k):
    n = len(arr)

    #initialize s1 or first center
    s = [0 for j in range(k)]
    #first center is first point
    s[0] = arr[0]

    phi = [0 for i in range(n)] #Assign All points to first center point


    for i in range(1,k):
        v = [0 for j in range(n)]
        Vtotal = 0
        for j in range(n):
            x = numpy.array(arr[j])
            center = numpy.array(s[i-1])
            v[j] = numpy.linalg.norm(x-center)**2
            Vtotal += v[j]
        probj = [0 for j in range(n)]
        for j in range(n):
            probj[j] = v[j]/Vtotal

        #Create Randomizer 
        randomizer = [0 for j in range(n)]
        randomizer[0] = probj[0]
        for j in range(1, n):
            randomizer[j] = randomizer[j-1]+probj[j]
            
        #Seed to choose next random center
        u = random.uniform(0.0, 1.0)

        #WWhat about the 0 - whatever range 
        #If number is betw
        if u > 0 and u <= randomizer[0]:
            s[i] = arr[0]
        for j in range(1,n):
            if u > randomizer[j-1] and u <= randomizer[j]:
                s[i] = arr[j]

        #After obtainingpoints run LLoyds
    
    #result = LloydAlgKarr(s, arr)

    return s

# Adjusts centers based on the closest points to the centers.
def LloydAlgArr(k, arr):
    n = len(arr)
    dim = len(arr[0])
    s = [0 for j in range(k)]
    groups = [[] for j in range(k)]

    for i in range(k):
        #s[i] = arr[random.randint(0, n)] #set random centers
        s[i] = arr[i] #set first 4 as centers
        groups[i] = []

    

    
    for z in range(100):
        # create groups of nearest points 
        for i in range(n):

            sVec = numpy.array(s[0])
            arrVec = numpy.array(arr[i])
            loc = 0

            closest = numpy.linalg.norm(sVec-arrVec)
            for j in range(1, k):
                sVec = numpy.array(s[j])
                distance = numpy.linalg.norm(sVec-arrVec)

                if distance < closest:
                    closest = distance
                    loc = j
        
            groups[loc].append(arr[i])

        # reassign centers
        for i in range(k):
            avg = [0 for m in range(dim)]
            size = len(groups[i])

            summations = [0 for w in range(dim)]
            for j in range(size):

                #Add all values in same column
                for l in range(dim):
                    summations[l] += groups[i][j][l]

            for j in range(dim):
                if size == 0:
                    avg[j] = 0
                else:
                    avg[j] = summations[j]/size


            s[i] = avg
        #repeat 20 times
    return s
            


    




# usees k++ to find starting center points to adjust later
def LloydAlgKarr(karr, arr):

    n = len(arr)
    k = len(karr)
    dim = len(arr[0])
    s = [0 for j in range(k)]
    groups = [[] for j in range(k)]

    for i in range(k):
        s[i] = karr[i] #set centers


    for z in range(100):
        # create groups of nearest points 
        for i in range(n):

            sVec = numpy.array(s[0])
            arrVec = numpy.array(arr[i])
            loc = 0

            closest = numpy.linalg.norm(sVec-arrVec)
            for j in range(1, k):
                sVec = numpy.array(s[j])
                distance = numpy.linalg.norm(sVec-arrVec)

                if distance < closest:
                    closest = distance
                    loc = j

            groups[loc].append(arr[i])

        # reassign centers
        for i in range(k):
            avg = [0 for m in range(dim)]
            size = len(groups[i])
            summations = [0 for w in range(dim)]
            for j in range(size):
                

                #Add all values in same column
                for l in range(dim):
                    summations[l] += groups[i][j][l]

            for j in range(dim):
                if size == 0:
                    avg[j] = 0
                else:
                    avg[j] = summations[j]/size

        
            s[i] = avg
            #repeat 20 times
    return s

def centerCost(arr, centers, phi):
    k = len(centers)
    maximum = 0

    for i in range(k):
        for j in range(len(phi)):
            if phi[j] == i:
                a = numpy.array(centers[i])
                b = numpy.array(arr[j])
                distance = numpy.linalg.norm(a-b)
                if distance > maximum:
                    maximum = distance
    return maximum

def meanCost(arr, centers):
    k = len(centers)
    summation = 0

    for j in range(len(arr)):
        a = numpy.array(centers[0])
        b = numpy.array(arr[j])
        minimim = numpy.linalg.norm(a-b)
        loc = 0
        for l in range(1, k):
            a = numpy.array(centers[l])
            b = numpy.array(arr[j])
            distance = numpy.linalg.norm(a-b)
            if(distance < minimim):
                minimim=distance
                loc = l
        summation += minimim**2
        
    average = numpy.sqrt(summation/len(arr))
    return average

def PlotAcumulativeDensity(data):
    m = len(data)

    values, base = numpy.histogram(data, bins=m)
    #evaluate the cumulative
    cumulative = numpy.cumsum(values)

    # plot the cumulative function
    X2 = numpy.sort(data)
    F2 = numpy.array(range(m))/float(m)

    #Label Graph
    plt.title('Cumulative Distribution of 4-Means Cost KMeans++')
    plt.xlabel('4-Mean Cost')
    plt.ylabel('Percentage Hit')

    plt.plot(X2, F2)

    plt.show()

def PlotAcumulativeDensityL(data):
    m = len(data)

    values, base = numpy.histogram(data, bins=m)
    #evaluate the cumulative
    cumulative = numpy.cumsum(values)

    # plot the cumulative function
    X2 = numpy.sort(data);
    F2 = numpy.array(range(m))/float(m)

    #Label Graph
    plt.title('Cumulative Distribution of 4-Means Cost Lloyd')
    plt.xlabel('4-Mean Cost')
    plt.ylabel('Percentage Hit')

    plt.plot(X2, F2)

    plt.show()

#Part 1
arr = textFileReadingAndStorage('C1.txt')
singleLinkClusters(arr)
completeLinkClusters(arr)
meanLinkClusters(arr)

#Part 2 A
arr = textFileReadingAndStorage('C2.txt')
centers = GonzalezAlgorthim(arr, 4)
global phiG
print(centers)
print(centerCost(arr, centers, phiG))
print(meanCost(arr, centers))
plotScatterForGonz(arr, centers, phiG)

# #Parrt 2 C.2
lcenters = LloydAlgKarr(centers, arr)
plotScatterForLloyd(arr, lcenters)
print(meanCost(arr, lcenters))


#Part 2 B
arr =textFileReadingAndStorage('C2.txt')
mean = [0 for i in range(100)]
lmean = [0 for i in range(100)]
for i in range(100):
    centers = KMeansPP(arr, 4)
    #lcenters = LloydAlgKarr(centers, arr)
    mean[i] = meanCost(arr, centers)
    #lmean[i] = meanCost(arr, lcenters)
print(centers)
print(lcenters)
PlotAcumulativeDensity(mean)


#Part 2 C.3
PlotAcumulativeDensityL(lmean)
hits = 0
size = len(mean)
for i in range(size):
    if mean[i]==lmean[i]:
        hits += 1
print(float(hits)/size)

#Part 3A

arr =textFileReadingAndStorage('C2.txt')
lcenters = LloydAlgArr(4, arr)

plotScatterForLloyd(arr, lcenters)
print(meanCost(arr, lcenters))







