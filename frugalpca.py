import numpy as np
import pandas as pd
from scipy.linalg import orthogonal_procrustes
from pandas_plink import read_plink
import dask.array as da
from dask import compute
from chest import Chest
import time
from datetime import datetime
from subprocess import call
import os.path


# def procrustes_old(data1, data2):
#     mtx1 = np.array(data1, dtype=np.double, copy=True)
#     mtx2 = np.array(data2, dtype=np.double, copy=True)

#     if mtx1.ndim != 2 or mtx2.ndim != 2:
#         raise ValueError("Input matrices must be two-dimensional")
#     if mtx1.shape != mtx2.shape:
#         raise ValueError("Input matrices must be of same shape")
#     if mtx1.size == 0:
#         raise ValueError("Input matrices must be >0 rows and >0 cols")

#     # translate all the data to the origin
#     mtx1_mean = np.mean(mtx1, 0)
#     mtx1 -= mtx1_mean
#     mtx2_mean = np.mean(mtx2, 0)
#     mtx2 -= mtx2_mean

#     # change scaling of data (in rows) such that trace(mtx*mtx') = 1
#     norm1 = np.linalg.norm(mtx1)
#     norm2 = np.linalg.norm(mtx2)
#     if norm1 == 0 or norm2 == 0:
#         raise ValueError("Input matrices must contain >1 unique points")
#     mtx1 /= norm1
#     mtx2 /= norm2

#     # transform mtx2 to minimize disparity
#     R, s = orthogonal_procrustes(mtx2, mtx1)
#     # orthogonal_procrustes can only find the best transformation between normalilzed matrices
#     s *= norm1 / norm2
#     b = mtx1_mean - mtx2_mean @ R * s

#     return R, s, b



DIM_REF = 4
DIM_STUDY = 20
DIM_STUDY_HIGH = DIM_STUDY * 2
DIM_ONLINESVD = DIM_STUDY * 2
DIM_RANDSVD = DIM_STUDY * 4
NITER_RANDSVD = 2

np.random.seed(21)

def eig_sym(XTX):
    ssq, V = np.linalg.eigh(XTX)
    V = np.squeeze(V)
    ssq = np.squeeze(ssq)
    s = np.sqrt(abs(ssq))
    V = V.T[::-1].T
    s = s[::-1]
    return s, V

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
    # TODO:Use eigendecomposition to speedup
    d2, R_Vt = np.linalg.svd(R, full_matrices=False)[1:] 
    V_new = np.zeros((k+1, n+1))
    V_new[:k, :n] = V1.transpose()
    V_new[k, n] = 1
    V2 = (R_Vt @ V_new).transpose()[:,:l]
    return d2, V2

def test_online_svd_procrust():
    # def test_svd_online():
    print("Testing test_svd_online...")

    # # For debugging only
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
    d2, PC_new = svd_online(U1, d1, V1, b, PC_new_dim)

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
    np.savetxt('test_PC_ref.dat', PC_ref)
    np.savetxt('test_PC_ref_fat.dat', PC_ref_fat)
    np.savetxt('test_PC_new_head.dat', PC_new_head)
    R, rho, c = procrustes(PC_ref_fat, PC_new_head)
    R_diffdim, rho_diffdim, c_diffdim = procrustes_diffdim(PC_ref, PC_new_head)
    # PC_new_tail_trsfed = PC_new_tail @ R * rho + c
    # PC_new_tail_trsfed = PC_new_tail_trsfed.flatten()[:PC_ref_dim]
    call(['make', 'procrustes.o'])
    call(['./procrustes.o'])
    R_trace = np.loadtxt('procrustes_A.dat')
    rho_trace = np.loadtxt('procrustes_rho.dat')
    c_trace = np.loadtxt('procrustes_c.dat')
    R_diffdim_trace = np.loadtxt('pprocrustes_A.dat')
    rho_diffdim_trace = np.loadtxt('pprocrustes_rho.dat')
    c_diffdim_trace = np.loadtxt('pprocrustes_c.dat')
    assert np.allclose(R_trace, R)
    assert np.allclose(rho_trace, rho)
    assert np.allclose(c_trace, c)
    assert np.allclose(R_diffdim_trace, R_diffdim)
    assert np.allclose(rho_diffdim_trace, rho_diffdim)
    assert np.allclose(c_diffdim_trace, c_diffdim)


    print("Passed!")

def procrustes(Y_mat, X_mat):
    ''' Find the best transformation from X to Y '''
    X = np.array(X_mat, dtype=np.double, copy=True)
    Y = np.array(Y_mat, dtype=np.double, copy=True)
    n_X = X.shape[0]
    X_mean = np.mean(X, 0)
    Y_mean = np.mean(Y, 0)
    X -= X_mean
    Y -= Y_mean
    C = Y.T @ X
    U, s, VT = np.linalg.svd(C, full_matrices=False)
    # TODO: Change to np.sum(X**2) for speed
    # trXX = np.trace(X.T @ X) 
    # trYY = np.trace(Y.T @ Y)
    trXX = np.sum(X**2)
    trYY = np.sum(Y**2)
    trS = np.sum(s)
    A = VT.T @ U.T
    rho = trS / trXX
    b = Y_mean - rho * X_mean @ A
    return A, rho, b


def procrustes_diffdim(Y_mat, X_mat, n_iter_max=int(1e4), epsilon_min=1e-6):
    X = np.array(X_mat, dtype=np.double, copy=True)
    Y = np.array(Y_mat, dtype=np.double, copy=True)
    n_X, p_X = X.shape
    n_Y, p_Y = Y.shape
    assert n_X == n_Y
    assert p_X >= p_Y
    if p_X == p_Y:
        R, rho, c = procrustes(Y, X)
    else:
        Z = np.zeros((n_X, p_X - p_Y))
        for i in range(n_iter_max):
            W = np.hstack((Y, Z))
            R, rho, c = procrustes(W, X)
            X_new = X @ R * rho + c
            Z_new = X_new[:, p_Y:]
            Z_new_mean = np.mean(Z_new, 0)
            Z_new_centered = Z_new - Z_new_mean
            Z_diff = Z_new - Z
            epsilon = np.sum(Z_diff**2) / np.sum(Z_new_centered**2)
            if(epsilon < epsilon_min):
                break
            else:
                Z = Z_new
    return R, rho, c

test_online_svd_procrust()

# # Read data
# if 'X' not in locals():
#     print("Reading reference data...")
#     X = read_plink('../data/kgn/kgn_chr_all_keep_orphans_snp_hgdp_biallelic_a2allele_train')[2]
#     X = X.astype(np.float32)
#     X = X.compute()
#     p_ref, n_ref = X.shape
#     print("Done.")

#     # Center and nomralize reference data
#     print("Centering and normalizing reference data...")
#     X_mean = np.nanmean(X, axis = 1).reshape((-1, 1))
#     X_std = np.nanstd(X, axis = 1).reshape((-1,1))
#     X_std[X_std == 0] = 1
#     X -= X_mean
#     X /= X_std
#     print("Done")

# if 'W' not in locals():
#     print("Reading study data...")
#     W = read_plink('../data/kgn/kgn_chr_all_keep_orphans_snp_hgdp_biallelic_a2allele_test')[2]
#     W_save = read_plink('../data/kgn/kgn_chr_all_keep_orphans_snp_hgdp_biallelic_a2allele_test')[2]
#     W = W.compute()
#     print("Done.")

#     # Center and nomralize study data
#     print("Centering and normalizing study data...")
#     W -= X_mean
#     W /= X_std
#     print("Done.")

# # PCA on the reference data
# if not ('s' in locals() and 'V' in locals()):
#     sV_file_all_exists = os.path.isfile('s.dat') and os.path.isfile('V.dat')
#     if sV_file_all_exists:
#         print("Reading existing s.dat and V.dat...")
#         s = np.loadtxt('s.dat')
#         V = np.loadtxt('V.dat')
#         print("Done.")
#     else:
#         print(datetime.now())
#         start_time = time.time()
#         # X = da.rechunk(X, (X.chunks[0], (X.shape[1])))
#         cache = Chest(path='cache')

#         # Direct svd
#         # print("Doing SVD on training data...")
#         # X = da.rechunk(X, (X.chunks[0], (X.shape[1])))
#         # U, s, Vt = da.linalg.svd(X)
#         # np.savetxt('U.dat', U, fmt='%10.5f')

#         # Compressed svd
#         # print("Doing randomized SVD on training data...")
#         # U, s, Vt = da.linalg.svd_compressed(X, DIM_RANDSVD, NITER_RANDSVD)
#         # U, s, Vt = compute(U, s, Vt, cache=cache)
#         # V = Vt.T
#         # np.savetxt('U.dat', U, fmt='%10.5f')

#         # Multiplication and eigendecomposition
#         print("Doing multiplication and eigendecomposition on training data...")
#         XTX = X.T @ X
#         np.savetxt('XTX.dat', XTX, fmt='%10.5f')
#         s, V = eig_sym(XTX)

#         elapse = time.time() - start_time
#         print(datetime.now())
#         print(elapse)
#         print("Saving training SVD result...")
#         np.savetxt('s.dat', s, fmt='%10.5f')
#         np.savetxt('V.dat', V, fmt='%10.5f')
#         print("Done.")

# # Subset and whiten PC scores
# print("Subsetting and whitening PC scores...")
# V = V[:, :DIM_STUDY_HIGH]
# s = s[:DIM_STUDY_HIGH]
# Vs = V * s
# pcs_ref = Vs[:, :DIM_REF]
# print("Done.")

# # This should be run only when mult&eigen is used.
# # Calculate PC loading
# print("Checking if PC loadings are calculated...")
# if 'U' in locals():
#     print("PC loadings matrix exists in memory.")
# else:
#     if os.path.isfile('U.dat'):
#         print("Reading existing U.dat...")
#         U = np.loadtxt('U.dat')
#     else:
#         print("Calculating PC loadings...")
#         U = X @ (V / s)
#     print("Done.")

# # if 'XTX' not in locals():
# #     XTX = np.loadtxt('XTX.dat')

# # Test result close to TRACE's
# print("Testing reference PC scores are the same as TRACE's...")
# pcs_ref_trace_file = '../data/kgn_kgn_1/kgn_chr_all_keep_orphans_snp_hgdp_biallelic_train_test.RefPC.coord'
# pcs_ref_trace = pd.read_table(pcs_ref_trace_file)
# pcs_ref_trace = np.array(pcs_ref_trace.iloc[:, 2:])
# ref_dim_trace = pcs_ref_trace.shape[1]
# for i in range(ref_dim_trace):
#     corr = np.corrcoef(pcs_ref_trace[:,i], pcs_ref[:,i])[0,1]
#     assert abs(corr) > 0.99
#     if corr < 0:
#         pcs_ref[:,i] *= -1
#     if(i < ref_dim_trace // 2):
#         assert np.allclose(pcs_ref_trace[:,i], pcs_ref[:,i], 0.01, 0.05)
# print("Passed.")

# print("Testing study reference PC scores are the same as TRACE's...")
# print("Calculating XTX_new...")
# print(datetime.now())
# b = W[:,0].reshape((1,-1))
# bX = b @ X
# bb = b @ b.T
# XTX_new = np.vstack((np.hstack((XTX, bX.T)), np.hstack((bX, bb))))
# print("Calculating s_new and V_new...")
# print(datetime.now())
# s_new, V_new = eig_sym(XTX_new)
# Vs_new = V_new * s_new
# pcs_new = Vs_new[:, :DIM_STUDY]
# print("Done.")
# print(datetime.now())

# print("Procrustes analysis...")
# print(datetime.now())
# pcs_new_head, pcs_new_tail = pcs_new[:-1, :], pcs_new[-1, :].reshape((1,-1))
# pcs_ref_fat = np.zeros((n_ref, DIM_STUDY))
# pcs_ref_fat[:, :DIM_REF] = pcs_ref
# R, rho, c = procrustes(pcs_ref_fat, pcs_new_head)
# pcs_new_tail_trsfed = pcs_new_tail @ R * rho + c
# pcs_new_tail_trsfed = pcs_new_tail_trsfed.flatten()[:DIM_REF]
# print(datetime.now())
# print("Done.")

# print("Check results.")

# # print("Calculating study PC scores with online SVD...")
# # s2, V2 = svd_online(U, s, V, b, DIM_ONLINESVD)
# # V2 = V2[:, :DIM_STUDY]
# # s2 = s2[:DIM_STUDY]
# # Vs2 = V2 * s2
# # Vs2_head = Vs2[:-1, :]
# # Vs2_tail = Vs2[-1, :]
# # pcs_ref_fat = np.concatenate((pcs_ref, np.zeros((n_ref, DIM_STUDY - DIM_REF))), axis=1)
# # R, rho, c = procrustes(pcs_ref_fat, Vs2_head)
# # Vs2_tail_mapped = Vs2_tail @ R * rho + c

