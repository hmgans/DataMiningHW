#This is the python file for AS01
from random import random
from random import randint
import numpy as np
import matplotlib.pyplot as plt
import time


#Part 1A
# n - int that is the max of the domain 1-n
def randomIntCollision(n):

    trialNumber = 0;
    dictionary = dict();

    while(True):
        trialNumber+=1;
        randomInt = randint(1, n); #inclusive so if n = 3000, [1,3000]
        if(dictionary.get(randomInt) == 1): #Collision detected
            return trialNumber;
        dictionary[randomInt] = 1;

#Part1B
# Creates an array with the number of random ints needed to be
# created for a collision for each trial.
# m - number of trials
# n - domain size
def randomCollisionsArr(m, n):
    start = 0;
    list =[] # list of

    for x in range(start, m):
        list.append(randomIntCollision(n));
    return list;

#1B Creates a cummulative density plot for the randomCollisionsArr generated.
def createPlotFor1B(m, n):
    # generate m collision trials
    data = randomCollisionsArr(m, n);
    # evaluate the histogram
    values, base = np.histogram(data, bins=m);
    #evaluate the cumulative
    cumulative = np.cumsum(values);

    # plot the cumulative function
    X2 = np.sort(data);
    F2 = np.array(range(m))/float(m);

    #Label Graph
    plt.title('Cumulative Distribution of Collisions');
    plt.xlabel('Number of Trials Until Collision');
    plt.ylabel('Percentage Hit');

    plt.plot(X2, F2)

    plt.show()

# This gives an average of the number of numbers generated to get a collision
# m - number of trials
# n - domain size
def empiricalEstimate1C(m, n):
    data = randomCollisionsArr(m, n);
    sum = 0;
    for x in range(0, m):
        sum += data[x];
    return sum/m;

# This method gives the average time to calculate random collisions for varying m and n
# m - number of trials
# n - domain
def timingCollision(m, n):

    timesToLoop=3;
    startTime = time.time();
    for x in range(0,timesToLoop):
        randomCollisionsArr(m,n);

    midpointTime = time.time();

    #Run empty loop for accuracy
    for x in range(0,timesToLoop):
        pass;
    stopTime = time.time();

    # compute average time of timesToLoop
    averageTime = ((midpointTime - startTime) - (stopTime-midpointTime))/timesToLoop;
    return averageTime;

#This creates a graph for varying m in relation to time in seconds.
def graphingForVaryingM(m):
    timingData = [];
    nValues = [];
    for n in range(250000, 1000001, 250000):
        timingData.append(timingCollision(m,n));
        nValues.append(n);

    plt.title("M = "+str(m));
    plt.xlabel("Value of N");
    plt.ylabel("Time Until Colision in Seconds");
    plt.plot(nValues, timingData);
    plt.show()

################################################################################################
#COUPON COLLECTORS
#Part 2A
# n - int that is the max of the domain 1-n
def randomIntCollisionForAll(n):
    trialNumber = 0;
    dictionary = dict();
    totalCollisions = 0;
    for x in range(1,n):
        dictionary[x] = False; # initialize to not visited

    while(totalCollisions != n-1):
        trialNumber+=1; #K
        randomInt = randint(1, n); #inclusive so if n = 3000, [1,3000] Domain size of 3000
        if(dictionary.get(randomInt) == False): # It has not been visited
            dictionary[randomInt] = True; # Set to true to indicate it has been visited
            totalCollisions += 1;
    return trialNumber;

#Part2B
# Creates an array with the number of random ints needed to be
# created for a all numbers of the domain to experience a collision.
# m - number of trials
# n - domain size
def randomCollisionsArrForCouponCollectors(m, n):
    start = 0;
    list =[] # list of trial numbers

    for x in range(start, m):
        list.append(randomIntCollisionForAll(n));
    return list;

#2B Creates a cummulative density plot for the randomCollisionsArr generated.
def createPlotFor2B(m, n):
    # generate m collision trials
    data = randomCollisionsArrForCouponCollectors(m, n);
    # evaluate the histogram
    values, base = np.histogram(data, bins=m);
    #evaluate the cumulative
    cumulative = np.cumsum(values);

    # plot the cumulative function
    X2 = np.sort(data);
    F2 = np.array(range(m))/float(m);

    #Label Graph
    plt.title('Cumulative Distribution of Collisions');
    plt.xlabel('Number of Trials Until All Slots Have a Collision');
    plt.ylabel('Percentage Hit');

    plt.plot(X2, F2)

    plt.show()

# This gives an average of the number of numbers generated so all
# numbers in the domain havee a collision
# m - number of trials
# n - domain size
def empiricalEstimate2C(m, n):
    data = randomCollisionsArrForCouponCollectors(m, n);
    sum = 0;
    for x in range(0, m):
        sum += data[x];
    return sum/m;


# This method gives the average time to calculate random collisions for varying m and n
# m - number of trials
# n - domain
def timingCollision2(m, n):

    timesToLoop=1;
    startTime = time.time();
    for x in range(0,timesToLoop):
        randomCollisionsArrForCouponCollectors(m,n);

    midpointTime = time.time();

    #Run empty loop for accuracy
    for x in range(0,timesToLoop):
        pass;
    stopTime = time.time();

    # compute average time of timesToLoop
    averageTime = ((midpointTime - startTime) - (stopTime-midpointTime))/timesToLoop;
    return averageTime;

#This creates a graph for varying m in relation to time in seconds.
def graphingForVaryingM2(m):
    timingData = [];
    nValues = [];
    for n in range(1000, 10001, 1000):
        timingData.append(timingCollision2(m,n));
        nValues.append(n);

    plt.title("M = "+str(m));
    plt.xlabel("Value of N");
    plt.ylabel("Time Until All N have a Colision in Seconds");
    plt.plot(nValues, timingData);
    plt.show()


#print(randomIntCollision(3000));
#createPlotFor1B(200, 3000);
#print(empiricalEstimate1C(200, 3000));

#Timing Section
#graphingForVaryingM(200);
#graphingForVaryingM(5000);
#graphingForVaryingM(10000);
################################################################################
#Next part

#print(randomIntCollisionForAll(200));
#print(empiricalEstimate2C(300, 200));

#Timing Section
#graphingForVaryingM2(300);
#graphingForVaryingM2(1000);
#graphingForVaryingM2(2000);
