import numpy as np


def svd_online(U1, d1, V1, b, k, l):
    assert(l <= k)
    p = U1.shape[0]
    b = b.reshape((p,1)) # Make sure the new sample is a column vec
    n, k = V1.shape
    b_tilde = b - U1 @ (U1.transpose() @ b)
    b_tilde = b_tilde / np.sqrt(sum(np.square(b_tilde)))
    R = np.concatenate((np.diag(d1), U1.transpose() @ b), axis = 1)
    R_tail = np.concatenate((np.zeros((1,k)), b_tilde.transpose() @ b), axis = 1)
    R = np.concatenate((R, R_tail), axis = 0)
    R_Vt = np.linalg.svd(R, full_matrices=False)[2]
    V_new = np.zeros((k+1, n+1))
    V_new[:k, :n] = V1.transpose()
    V_new[k, n] = 1
    PC = (R_Vt @ V_new).transpose()[:,:l]
    return PC

def test_svd_online():
    print("Testing test_svd_online...")

    # # For comparing with the R script written by Shawn
    # X = np.loadtxt('test_X.dat', delimiter='\t')
    # b = np.loadtxt('test_b.dat', delimiter='\t')

    # Generate testing matrices
    p = 1000
    n = 200
    X = np.random.normal(size = p * n).reshape((p,n))
    b = np.random.normal(size = p).reshape((p,1))

    # Parameters for onlineSVD
    k = 100 # Number of PC's calculated by online SVD
    l = 4 # Number of PC's we want

    # Decompose the training matrix
    U, d, Vt = np.linalg.svd(X, full_matrices = False)
    V = Vt.transpose()
    # Subset the PC scores since we only need the first k PC's
    U1 = U[:, :k]
    d1 = d[:k]
    V1 = V[:, :k]
    PC = svd_online(U1, d1, V1, b, k, l)

    # Test if the result is close enough
    trueAns = np.linalg.svd(np.concatenate((X,b),axis=1))[2].transpose()[:,:l]
    for i in range(l):
        assert \
            abs(np.max(PC[:,i] - trueAns[:,i])) < 0.01 or \
            abs(np.max(PC[:,i] + trueAns[:,i])) < 0.01 # online_svd can flip the sign of a PC
    print("Passed!")

test_svd_online()



