#librery
import numpy as np
from numpy.linalg import eig

def matrix_a(row,col):
    #Create a random matrix
    row = int(row)
    col = int(col)
    np.random.seed(12334)
    matrix= np.random.randint(10, size=(row, col))
    np.random.seed(1234)
    print(matrix)
    #What is the rank and trace of A?
    rank =  np.linalg.matrix_rank(matrix)
    trace = np.trace(matrix)
    #Can you invert A? How?
    #First calculate det of matrix
    try:
        det=np.linalg.det(matrix)
        invert= np.linalg.inv(matrix)
        print(det,invert)
    except:
        det="det diferent 0"
        invert="This matrix is not invertible"
    try:
        #eigenvalues and eigenvectors of A'A
        invert= np.linalg.inv(matrix)
        A = invert*matrix
        w_1,v_1=eig(A)
        #eigenvalues and eigenvectors of AA'
        A = matrix*invert
        w_2,v_2=eig(A)
        #We can notice that the results of eigenvalues and eigenvectors of A'A and AA' are the same
        return (matrix, int(rank), int(trace), det,invert, w_1, v_1, w_2, v_2)
    except:
        w_1=""
        v_1=""
        w_2=""
        v_2=""
        return (matrix, int(rank), int(trace), det,invert, w_1, v_1, w_2, v_2)
