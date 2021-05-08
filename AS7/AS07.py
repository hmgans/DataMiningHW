import numpy as np
from scipy.spatial import distance_matrix
from scipy import linalg as LA
import matplotlib.pyplot as plt



def SVDLNorm(k):
    A = np.loadtxt('A.csv', delimiter=',')
    U, s, Vt = LA.svd(A, full_matrices=False)

    S = np.diag(s)

    for i in range(1,k+1):
        Uk = U[:,:i]
        Sk = S[:i,:i]
        Vtk = Vt[:i,:]
        Ak = Uk @ Sk @ Vtk
        print(LA.norm(A-Ak,2))

    print(LA.norm(A, 2)*.2)

def Ak(k):
    A = np.loadtxt('A.csv', delimiter=',')
    U, s, Vt = LA.svd(A, full_matrices=False)

    S = np.diag(s)

    Uk = U[:,:k]
    Sk = S[:k,:k]
    Vtk = Vt[:k,:]
    Ak = Uk @ Sk @ Vtk
    return Ak

def Plot2D():

    A = np.loadtxt('A.csv', delimiter=',')
    D = distance_matrix(A,A)
    print(len(D[0]))

    D = np.array(D)
    D2 = np.square(D)

    n = 3500 # starting size

    #double centering
    C = np.eye(n) - (1/n)*np.ones(n)
    M = -0.5 * C @ D2 @ C
    #eigendecomposition (forcing real eigenvalue squaroots)
    print("Start eig")
    l,V = LA.eig(M)
    print("End eig")


    s = np.real(np.power(l,0.5))
    print("done with reals")
    V2 = V[:,[0,1]]
    s2 = np.diag(s[0:2])
    #low (2) dimensiona points
    Q = V2 @ s2
    #printing

    plt.plot(Q[:,0],Q[:,1],'ro')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Dim Reduce to 2 Dimenstions')
    plt.show()

def freq_dir(A,l):
    r = A.shape[0]
    c = A.shape[1]
    B = np.zeros([l*2, c])
    B[:l-1, :] = A[:l-1,:]
    zerorows = l + 1
    for i in range(l-1,r):
      """
      implement the algorithm 16.2.1 in L16 MatrixSketching in Data Mining course webpage
          insert ith row into a zero-value row of the mat_b
          
          if B has no zero-valued rows ( can be ketp track with counter) then:
            U,S,V = svd(mat_b)  using  U,S,V = np.linalg.svd(mat_b,full_matrices = False)
            ...
            procedure same as the algorithm 16.2.1
            ...
      """
      #Get zero row
      zero_rows = np.where(~(B).any(axis=1))[0]
      if len(zero_rows) > 1:
          B[zero_rows[0], :] = A[i,:]
      else:
          B[zero_rows[0], :] = A[i,:]
          U, S, V = LA.svd(B, full_matrices=False)

          S_prime = np.zeros(S.shape)
          # Do calculations
          S_prime[0: l-1]= np.power(S[:l-1]**2 - S[l-1]**2, .5)
          # Get Diagonals
          S_prime = np.diag(S_prime)

          # Recalculate B
          B = S_prime @ V
        

    return B


def part2():
    A = np.loadtxt('A.csv', delimiter=',')
    B = freq_dir(A, 9)
    
    U, S, V = LA.svd(B, full_matrices=False)
    print(S)


    #print('A_F: ' + str(LA.norm(A, 'fro')**2/20))

    print('A-Ak_F: ' + str(LA.norm(A - Ak(2), 'fro')**2/20))

    
    print('Error:' + str(LA.norm((A.T @ A - B.T @ B),2)))



# K = 10
#SVDLNorm(10)
#Plot2D()

part2()






