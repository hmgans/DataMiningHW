### Loading Data
#-------------------------------------------------------------------------------
# import packages
import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as la




#-------------------------------------------------------------------------------
### Linear Regression
#-------------------------------------------------------------------------------
# Add a column of ones to account for the bias in the data
#bias = np.ones(shape = (len(X),1))
#X = np.concatenate((bias, X), 1)
# Matrix form of linear regression
def part1():
    # Load data from files as numpy arrays 
    X = np.loadtxt('X.csv', delimiter=',')
    y = np.loadtxt('y.csv', delimiter=',')
    print(len(y))


    alpha = la.inv(X.T @ X) @ X.T @ y.T

    yHat = np.linalg.norm(y - X @ alpha, 2)
    print(yHat)

    ### Ridge Regression based solely off precipitation
    # -------------------------------------------------------------------------------
    # Add a column of ones to account for the bias in the data

    #bias = np.ones(shape = (len(X),1))
    #X = np.concatenate((bias, X), 1)
    I = np.identity(50)
    s = 0.1

    # Matrix form of linear regression
    alpha = la.inv((X.T @ X) + s * I) @ X.T @ y.T

    yHat = np.linalg.norm(y - X @ alpha, 2)
    print("s = 0.1")
    print(yHat)

    s = 0.3

    # Matrix form of linear regression
    alpha = la.inv((X.T @ X) + s * I) @ X.T @ y.T

    yHat = np.linalg.norm(y - X @ alpha, 2)
    print("s = 0.3")
    print(yHat)

    s = 0.7

    # Matrix form of linear regression
    alpha = la.inv((X.T @ X) + s * I) @ X.T @ y.T

    yHat = np.linalg.norm(y - X @ alpha, 2)
    print("s = 0.7")
    print(yHat)

    s = 0.9

    # Matrix form of linear regression
    alpha = la.inv((X.T @ X) + s * I) @ X.T @ y.T

    yHat = np.linalg.norm(y - X @ alpha, 2)
    print("s = 0.9")
    print(yHat)


    s = 1.1

    # Matrix form of linear regression
    alpha = la.inv((X.T @ X) + s * I) @ X.T @ y.T

    yHat = np.linalg.norm(y - X @ alpha, 2)
    print("s = 1.1")
    print(yHat)


    s = 1.3

    # Matrix form of linear regression
    alpha = la.inv((X.T @ X) + s * I) @ X.T @ y.T

    yHat = np.linalg.norm(y - X @ alpha, 2)
    print("s = 1.3")
    print(yHat)

    s = 1.5

    # Matrix form of linear regression
    alpha = la.inv((X.T @ X) + s * I) @ X.T @ y.T

    yHat = np.linalg.norm(y - X @ alpha, 2)
    print("s = 1.5")
    print(yHat)



def part2():

    # Load data from files as numpy arrays 
    X = np.loadtxt('X.csv', delimiter=',')
    y = np.loadtxt('y.csv', delimiter=',')

    #Different sections
    #X1-------------------TRAIN: 75 First, TEST: 25 Last-----------------------
    # Xr = X[75:,:]
    # yr = y[75:]
    # X = X[:75,:]
    # y = y[:75]
    # #X2-------------------TRAIN: 75 Last, TEST: 25 First-----------------------
    # Xr = X[:25,:]
    # yr = y[:25]
    # X = X[25:,:]
    # y = y[25:]
    # #X3-------------------TRAIN: 50 First 25 Last, TEST: 50-75-----------------
    # Xr = X[50:75,:]
    # yr = y[50:75]
    # X = np.vstack((X[:50,:], X[75:,:]))
    # y = np.concatenate((y[:50], y[75:]))
    #X4-------------------TRAIN: 25 First 50 Last, TEST: 25-50-----------------
    Xr = X[25:50,:]
    yr = y[25:50]
    X = np.vstack((X[:25,:], X[50:,:]))
    y = np.concatenate((y[:25], y[50:]))



    

    alpha = la.inv(X.T @ X) @ X.T @ y.T

    yHat = np.linalg.norm(yr - Xr @ alpha, 2)
    print(yHat)

    ### Ridge Regression based solely off precipitation
    # -------------------------------------------------------------------------------
    # Add a column of ones to account for the bias in the data

    #bias = np.ones(shape = (len(X),1))
    #X = np.concatenate((bias, X), 1)
    I = np.identity(50)
    s = 0.1

    # Matrix form of linear regression
    alpha = la.inv((X.T @ X) + s * I) @ X.T @ y.T

    yHat = np.linalg.norm(yr - Xr @ alpha, 2)
    print("s = 0.1")
    print(yHat)

    s = 0.3

    # Matrix form of linear regression
    alpha = la.inv((X.T @ X) + s * I) @ X.T @ y.T

    yHat = np.linalg.norm(yr - Xr @ alpha, 2)
    print("s = 0.3")
    print(yHat)

    s = 0.7

    # Matrix form of linear regression
    alpha = la.inv((X.T @ X) + s * I) @ X.T @ y.T

    yHat = np.linalg.norm(yr - Xr @ alpha, 2)
    print("s = 0.7")
    print(yHat)

    s = 0.9

    # Matrix form of linear regression
    alpha = la.inv((X.T @ X) + s * I) @ X.T @ y.T

    yHat = np.linalg.norm(yr - Xr @ alpha, 2)
    print("s = 0.9")
    print(yHat)


    s = 1.1

    # Matrix form of linear regression
    alpha = la.inv((X.T @ X) + s * I) @ X.T @ y.T

    yHat = np.linalg.norm(yr - Xr @ alpha, 2)
    print("s = 1.1")
    print(yHat)


    s = 1.3

    # Matrix form of linear regression
    alpha = la.inv((X.T @ X) + s * I) @ X.T @ y.T

    yHat = np.linalg.norm(yr - Xr @ alpha, 2)
    print("s = 1.3")
    print(yHat)

    s = 1.5

    # Matrix form of linear regression
    alpha = la.inv((X.T @ X) + s * I) @ X.T @ y.T

    yHat = np.linalg.norm(yr - Xr @ alpha, 2)
    print("s = 1.5")
    print(yHat)

part1()
#part2()