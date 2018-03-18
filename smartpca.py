import numpy as np
from scipy.linalg import orthogonal_procrustes

np.random.seed(21)

def svd_online(U1, d1, V1, b, l):
    n, k = V1.shape
    assert(l <= k)
    p = U1.shape[0]
    b = b.reshape((p,1)) # Make sure the new sample is a column vec
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


def procrustes(data1, data2):
    ''' This is a simple modification of scipy.spatial.procrustes.
    For our purpose, we need to get the rotation matrix and the scaling factor,
    but these two values are not returned in the original function.
    '''
    mtx1 = np.array(data1, dtype=np.double, copy=True)
    mtx2 = np.array(data2, dtype=np.double, copy=True)

    if mtx1.ndim != 2 or mtx2.ndim != 2:
        raise ValueError("Input matrices must be two-dimensional")
    if mtx1.shape != mtx2.shape:
        raise ValueError("Input matrices must be of same shape")
    if mtx1.size == 0:
        raise ValueError("Input matrices must be >0 rows and >0 cols")

    # translate all the data to the origin
    mtx1_mean = np.mean(mtx1, 0)
    mtx1 -= mtx1_mean
    mtx2_mean = np.mean(mtx2, 0)
    mtx2 -= mtx2_mean

    norm1 = np.linalg.norm(mtx1)
    norm2 = np.linalg.norm(mtx2)

    if norm1 == 0 or norm2 == 0:
        raise ValueError("Input matrices must contain >1 unique points")

    # change scaling of data (in rows) such that trace(mtx*mtx') = 1
    mtx1 /= norm1
    mtx2 /= norm2

    # transform mtx2 to minimize disparity
    R, s = orthogonal_procrustes(mtx2, mtx1)
    # orthogonal_procrustes can only find the best transformation between normalilzed matrices
    s *= norm1 / norm2
    b = mtx1_mean - mtx2_mean @ R * s

    return R, s, b

# def test_svd_online():
print("Testing test_svd_online...")

# # For comparing with the R script written by Shawn
# X = np.loadtxt('test_X.dat')
# b = np.loadtxt('test_b.dat').reshape((-1,1))
# p, n = X.shape

# Generate testing matrices
p = 1000
n = 200
X = np.random.normal(size = p * n).reshape((p,n))
b = np.random.normal(size = p).reshape((p,1))
np.savetxt('test_X.dat', X)
np.savetxt('test_b.dat', b)


# Center reference data
X_mean = np.mean(X, axis = 1).reshape((p,1))
X -= X_mean
# Nonrmalize referencd data
X_norm = np.std(X, axis = 1).reshape((p,1))
X_norm[X_norm == 0] = 1
X /= X_norm

# Center study data
b -= X_mean
b /= X_norm

# Parameters for onlineSVD
svd_online_dim = 100 # Number of PC's calculated by online SVD
PC_new_dim = 20 # Number of PC's we want for each new sample
PC_ref_dim = 4 # Number of PC's for the reference group
assert PC_new_dim <= svd_online_dim
assert PC_ref_dim <= PC_new_dim

# Decompose the training matrix
U, d, Vt = np.linalg.svd(X, full_matrices = False)
V = Vt.transpose()
PC_ref = V[:, :PC_ref_dim]
# Subset the PC scores since we only need the first k PC's
U1 = U[:, :svd_online_dim]
d1 = d[:svd_online_dim]
V1 = V[:, :svd_online_dim]
PC_new = svd_online(U1, d1, V1, b, PC_new_dim)

# Test if the result is close enough
trueAns = np.linalg.svd(np.concatenate((X,b),axis=1))[2].transpose()[:,:PC_new_dim]
for i in range(trueAns.shape[1]):
    assert \
        abs(np.max(PC_new[:,i] - trueAns[:,i])) < 0.05 or \
        abs(np.max(PC_new[:,i] + trueAns[:,i])) < 0.05 # online_svd can flip the sign of a PC
print("Passed!")

print("Testing procrustes...")

PC_new_head, PC_new_tail = PC_new[:-1, :], PC_new[-1, :].reshape((1,PC_new_dim))
PC_ref_fat = np.zeros(n * PC_new_dim).reshape((n, PC_new_dim))
PC_ref_fat[:, :PC_ref_dim] = PC_ref
np.savetxt('test_PC_ref_fat.dat', PC_ref_fat)
np.savetxt('test_PC_new_head.dat', PC_new_head)
R, s, c = procrustes(PC_ref_fat, PC_new_head)
PC_new_tail_trsfed = PC_new_tail @ R * s + c
PC_new_tail_trsfed = PC_new_tail_trsfed.flatten()[:PC_ref_dim]

print("Nothing is tested.")

# test_svd_online()


