import numpy as np

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
      zero_row = np.where(~(B).any(axis=1))[0]
      print(zero_row)

      

    return B
