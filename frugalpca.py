import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from pandas_plink import read_plink
import dask.array as da
# from dask import compute
# from chest import Chest
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from rpy2.robjects import numpy2ri
r = robjects.r
robjects.numpy2ri.activate()
importr('hdpca')
pc_adjust = r['pc_adjust']
import matplotlib
matplotlib.use('Agg') # For outputting plots in png on servers
import matplotlib.pyplot as plt
import time
from datetime import datetime
import os.path
from tempfile import mkdtemp
import subprocess
import sys
import logging


# print(datetime.now())
# print('Loading reference dask array into memmap...')
# W_memmap_filename = os.path.join(TMP_DIR, 'W_memmap.dat')
# W = np.memmap(W_memmap_filename, dtype=np.float32, mode='w+', shape=W_dask.shape)
# # W[:] = W_dask
# W = np.zeros(dtype=np.float32, shape=W_dask.shape)
# for i in range(len(W_dask.chunks[0])):
#     start = sum(W_dask.chunks[0][:i])
#     end = sum(W_dask.chunks[0][:i+1])
#     W[start:end, :] = W_dask[start:end, :]
# TODO: Add checking for ind-major vs snp-major


# print('Sorting ref snps by chrom and pos...')
# print(datetime.now())
# X_bim_full['chrom'] = X_bim_full['chrom'].astype(np.int64)
# W_bim_full['chrom'] = W_bim_full['chrom'].astype(np.int64)
# # X_bim_full = X_bim_full.sort_values(by=['chrom', 'pos'])
# print('Sorting stu snps by chrom and pos...')
# print(datetime.now())
# # W_bim_full = W_bim_full.sort_values(by=['chrom', 'pos'])
# print('Intersecting snps...')
# print(datetime.now())
# snp_intersect = np.intersect1d(X_bim_full['snp'], W_bim_full['snp'], assume_unique=True)
# print('Filtering reference snps...')
# print(datetime.now())
# X_snp_isshared = np.isin(X_bim_full['snp'], snp_intersect, assume_unique=True)
# print('Filtering study snps...')
# print(datetime.now())
# W_snp_isshared = np.isin(W_bim_full['snp'], snp_intersect, assume_unique=True)
# print('Creating filtered reference set...')
# print(datetime.now())
# X = X_full[X_snp_isshared].compute()
# X_bim = X_bim_full[X_snp_isshared]
# print('Creating filtered study set...')
# print(datetime.now())
# W_dask = W_full[W_snp_isshared]
# W_memmap_filename = os.path.join(mkdtemp(), 'W_memmap.dat')
# W = np.memmap(W_memmap_filename, dtype=np.float32, mode='w+', shape=W_dask.shape)
# W[:] = W_dask
# W_bim = W_bim_full[W_snp_isshared]

# print('Handling alleles...')
# allele_isdiff = np.array(W_bim['a0']) != np.array(X_bim['a0'])
# # allele_isswapped = np.logical_and(np.array(W_bim['a0']) == np.array(X_bim['a1']), np.array(W_bim['a0']) == np.array(X_bim['a1']))
# c0 = np.zeros((p, 1))
# c0[allele_isdiff] = 2
# c1 = np.ones((p, 1))
# c1[allele_isdiff] = -1
# W *= c1
# W += c0
# # TODO: Change the bim file, too
# # W_bim_a0_diff = W_bim['a0'][allele_isdiff]
# # W_bim['a0'][allele_isdiff] = W_bim['a1'][allele_isdiff].astype('object').values
# # W_bim['a1'][allele_isdiff] = W_bim_a0_diff
# print('Done.')
# print(datetime.now())

# print('Calculating pc scores with eigen decomposition...')
# print(datetime.now())
# if 'XTX' not in locals():
#     XTX = np.loadtxt('XTX.dat')
# pcs_stu_eig = np.zeros((4, DIM_REF))
# XTX_new = np.zeros((n_ref + 1, n_ref + 1))
# XTX_new[:-1, :-1] = XTX
# for i in range(4):
#     # print('Calculating XTX_new...')
#     # print(datetime.now())
#     b = W[:,i]
#     bX = b @ X
#     bb = np.sum(b**2)
#     XTX_new[-1, :-1] = bX
#     XTX_new[:-1, -1] = bX
#     XTX_new[-1, -1] = bb
#     # print('Calculating s_new and V_new...')
#     # print(datetime.now())
#     s_new, V_new = svd_eigcov(XTX_new)
#     Vs_new = V_new * s_new
#     pcs_new = Vs_new[:, :DIM_STUDY]
#     # print('Done.')
#     # print(datetime.now())
#     # print('Procrustes analysis...')
#     # print(datetime.now())
#     pcs_new_head, pcs_new_tail = pcs_new[:-1, :], pcs_new[-1, :].reshape((1,-1))
#     R, rho, c = procrustes_diffdim(pcs_ref, pcs_new_head)
#     pcs_new_tail_trsfed = pcs_new_tail @ R * rho + c
#     pcs_stu_eig[i, :] = pcs_new_tail_trsfed.flatten()[:DIM_REF]
# print('Done.')
# assert np.allclose(pcs_stu_trace[:4,:dim_stu_trace], pcs_stu_eig[:4,:dim_stu_trace], 0.01, 0.05)

# def procrustes_old(data1, data2):
#     mtx1 = np.array(data1, dtype=np.double, copy=True)
#     mtx2 = np.array(data2, dtype=np.double, copy=True)
#     if mtx1.ndim != 2 or mtx2.ndim != 2:
#         raise ValueError('Input matrices must be two-dimensional')
#     if mtx1.shape != mtx2.shape:
#         raise ValueError('Input matrices must be of same shape')
#     if mtx1.size == 0:
#         raise ValueError('Input matrices must be >0 rows and >0 cols')
#     # translate all the data to the origin
#     mtx1_mean = np.mean(mtx1, 0)
#     mtx1 -= mtx1_mean
#     mtx2_mean = np.mean(mtx2, 0)
#     mtx2 -= mtx2_mean
#     # change scaling of data (in rows) such that trace(mtx*mtx') = 1
#     norm1 = np.linalg.norm(mtx1)
#     norm2 = np.linalg.norm(mtx2)
#     if norm1 == 0 or norm2 == 0:
#         raise ValueError('Input matrices must contain >1 unique points')
#     mtx1 /= norm1
#     mtx2 /= norm2
#     # transform mtx2 to minimize disparity
#     R, s = orthogonal_procrustes(mtx2, mtx1)
#     # orthogonal_procrustes can only find the best transformation between normalilzed matrices
#     s *= norm1 / norm2
#     b = mtx1_mean - mtx2_mean @ R * s
#     return R, s, b

# print('Testing study PC scores are the same as TRACE's...')
# pcs_stu_trace_file = '../data/kgn_kgn_1/kgn_chr_all_keep_orphans_snp_hgdp_biallelic_train_test.ProPC.coord'
# pcs_stu_trace = pd.read_table(pcs_stu_trace_file)
# pcs_stu_trace = np.array(pcs_stu_trace.iloc[:, 6:])
# dim_stu_trace = pcs_stu_trace.shape[1]
# assert np.allclose(pcs_stu_trace, pcs_stu_onl, 0.01, 0.05)
# assert np.allclose(pcs_stu_trace, pcs_stu_onl, 0.01, 0.05)
# np.savetxt('pcs_stu_trace.dat', pcs_stu_trace, fmt=NP_OUTPUT_FMT, delimiter='\t')
# print('Passed.')

# PCA on the reference data
# sV_file_all_exists = os.path.isfile('s.dat') and os.path.isfile('V.dat')
# if sV_file_all_exists:
#     print('Reading existing s.dat and V.dat...')
#     s = np.loadtxt('s.dat')
#     V = np.loadtxt('V.dat')
#     print('Done.')
# X = da.rechunk(X, (X.chunks[0], (X.shape[1])))
# cache = Chest(path='cache')

# Compressed (randomized) svd
# print('Doing randomized SVD on training data...')
# U, s, Vt = da.linalg.svd_compressed(X, DIM_RANDSVD, NITER_RANDSVD)
# U, s, Vt = compute(U, s, Vt, cache=cache)
# V = Vt.T
# np.savetxt('U.dat', U, fmt=NP_OUTPUT_FMT)

# # Test reference pc scores close to TRACE's
# print('Testing reference PC scores are the same as TRACE's...')
# pcs_ref_trace_file = '../data/kgn_kgn_1/kgn_chr_all_keep_orphans_snp_hgdp_biallelic_train_test.RefPC.coord'
# pcs_ref_trace = pd.read_table(pcs_ref_trace_file)
# pcs_ref_trace = np.array(pcs_ref_trace.iloc[:, 2:])
# ref_dim_trace = pcs_ref_trace.shape[1]
# for i in range(ref_dim_trace):
#     corr = np.corrcoef(pcs_ref_trace[:,i], pcs_ref[:,i])[0,1]
#     assert abs(corr) > 0.99
#     if corr < 0:
#         pcs_ref[:,i] *= -1
#         V[:,i] *= -1
#     assert np.allclose(pcs_ref_trace[:,i], pcs_ref[:,i], 0.01, 0.05)
# print('Passed.')

# This should be run only when mult&eigen is used for decomposing reference data.
# This must be done after the signs of V are made to be same as TRACE's
# Calculate PC loading
# if os.path.isfile('U.dat'):
#     print('Reading existing U.dat...')
#     U = np.loadtxt('U.dat')

# def get_popu_ref_info(X_fam, popu_ref_filename, superpopu_ref_filename):
#     popu_ref_df = pd.read_table(popu_ref_filename)
#     popu_ref_df = popu_ref_df.rename(index=str, columns={'Individual ID' : 'iid'})
#     superpopu_ref_df = pd.read_table(superpopu_ref_filename)
#     superpopu_ref_dict = superpopu_ref_df.set_index('Population Code')['Super Population Code'].to_dict()
#     popu_ref_df['Superpopulation'] = [superpopu_ref_dict[popu_this] for popu_this in popu_ref_df['Population']]
#     indiv_ref_info = pd.merge(X_fam, popu_ref_df, on = 'iid')
#     return indiv_ref_info[['Population', 'Superpopulation']]

DIM_REF = 4
assert DIM_REF >= 4
DIM_STUDY = 20
PLOT_ALPHA_REF=0.05
PLOT_ALPHA_STU=0.7
N_NEIGHBORS=5

DIM_STUDY_HIGH = DIM_STUDY * 2
DIM_SVDONLINE = DIM_STUDY * 2
# DIM_SVDRAND = DIM_STUDY * 4
# NITER_SVDRAND = 2
HDPCA_N_SPIKE_MAX = 20
NP_OUTPUT_FMT = '%.4f'
DELIMITER = '\t'
TMP_DIR = mkdtemp()
CHUNK_SIZE_STUDY = 1000

def create_logger(prefix):
    log = logging.getLogger()
    log.handlers = [] # Avoid duplicated logs in interactive modes
    log.setLevel(logging.INFO)
    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    # create file handler which logs even debug messages
    filename = prefix + '.' + str(round(time.time())) + '.log'
    fh = logging.FileHandler(filename, 'w')
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    log.addHandler(fh)
    # create console handler with a higher log level
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    log.addHandler(ch)
    return log


def svd_eigcov(XTX):
    ssq, V = np.linalg.eigh(XTX)
    V = np.squeeze(V)
    ssq = np.squeeze(ssq)
    s = np.sqrt(abs(ssq))
    V = V.T[::-1].T
    s = s[::-1]
    return s, V

def svd_online(U1, d1, V1, b, l):
    n, k = V1.shape
    assert U1.shape[1] == k
    assert len(d1) == k
    assert(l <= k)
    p = U1.shape[0]
    b = b.reshape((p,1)) # Make sure the new sample is a column vec
    b_tilde = b - U1 @ (U1.T @ b)
    b_tilde = b_tilde / np.sqrt(sum(np.square(b_tilde)))
    R = np.concatenate((np.diag(d1), U1.transpose() @ b), axis = 1)
    R_tail = np.concatenate((np.zeros((1,k)), b_tilde.transpose() @ b), axis = 1)
    R = np.concatenate((R, R_tail), axis = 0)
    d2, R_Vt = np.linalg.svd(R, full_matrices=False)[1:] 
    # TODO: Try using chomsky decomposition on R to speed up
    # Since R is triangular
    # Eigencomposition's runtime is the same as svd's
    # d2, R_V = svd_eigcov(R.T @ R)
    # R_Vt = R_V.T
    V_new = np.zeros((k+1, n+1))
    V_new[:k, :n] = V1.transpose()
    V_new[k, n] = 1
    V2 = (R_Vt @ V_new).transpose()[:,:l]
    return d2, V2

def test_online_svd_procrust():
    np.random.seed(21)
    # def test_svd_online():
    logging.info('Testing test_svd_online...')

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
    logging.info('Passed!')

    logging.info('Testing procrustes...')
    PC_new_head, PC_new_tail = PC_new[:-1, :], PC_new[-1, :].reshape((1,PC_new_dim))
    PC_ref_fat = np.zeros(n * PC_new_dim).reshape((n, PC_new_dim))
    PC_ref_fat[:, :PC_ref_dim] = PC_ref
    np.savetxt('test_PC_ref.dat', PC_ref)
    np.savetxt('test_PC_ref_fat.dat', PC_ref_fat)
    np.savetxt('test_PC_new_head.dat', PC_new_head)
    # Test procrustes with the same dimension
    R, rho, c = procrustes(PC_ref_fat, PC_new_head)
    # PC_new_tail_trsfed = PC_new_tail @ R * rho + c
    # PC_new_tail_trsfed = PC_new_tail_trsfed.flatten()[:PC_ref_dim]
    subprocess.run(['make', 'procrustes.o'], stdout=subprocess.PIPE)
    subprocess.run(['./procrustes.o'], stdout=subprocess.PIPE)
    R_trace = np.loadtxt('procrustes_A.dat')
    rho_trace = np.loadtxt('procrustes_rho.dat')
    c_trace = np.loadtxt('procrustes_c.dat')
    assert np.allclose(R_trace, R)
    assert np.allclose(rho_trace, rho)
    assert np.allclose(c_trace, c)
    # Test procrustes with different dimensions
    R_diffdim, rho_diffdim, c_diffdim = procrustes_diffdim(PC_ref, PC_new_head)
    R_diffdim_trace = np.loadtxt('pprocrustes_A.dat')
    rho_diffdim_trace = np.loadtxt('pprocrustes_rho.dat')
    c_diffdim_trace = np.loadtxt('pprocrustes_c.dat')
    assert np.allclose(R_diffdim_trace, R_diffdim)
    assert np.allclose(rho_diffdim_trace, rho_diffdim)
    assert np.allclose(c_diffdim_trace, c_diffdim)
    logging.info('Passed!')

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

def intersect_ref_stu_snps(ref_pref, stu_pref):
    logging.info('Intersecting .bed files by using bash and plink...')
    bashout = subprocess.run(['bash', 'intersect_bed.sh', ref_pref, stu_pref, TMP_DIR], stdout=subprocess.PIPE)
    ref_pref_commsnpsrefal, stu_pref_commsnpsrefal = bashout.stdout.decode('utf-8').split('\n')[-3:-1]
    assert len(ref_pref_commsnpsrefal) > 0
    assert len(stu_pref_commsnpsrefal) > 0
    return ref_pref_commsnpsrefal, stu_pref_commsnpsrefal

def bed2dask(filename, X_dtype=np.float32):
    logging.info('Reading Plink binary data into dask array...')
    X_bim, X_fam, X_dask = read_plink(filename, verbose=False)
    X_dask = X_dask.astype(X_dtype)
    return X_dask, X_bim, X_fam 

def dask2memmap(X_dask, X_memmap_filename):
    logging.info('Loading dask array into memmap...')
    X_memmap_filename = os.path.join(TMP_DIR, X_memmap_filename)
    X = np.memmap(X_memmap_filename, dtype=np.float32, mode='w+', shape=X_dask.shape)
    X[:] = X_dask
    return X

def standardize_ref(X):
    logging.info('Centering, normalizing, and imputing reference data...')
    X_mean = np.nanmean(X, axis = 1).reshape((-1, 1))
    X_std = np.nanstd(X, axis = 1).reshape((-1,1))
    X_std[X_std == 0] = 1
    X -= X_mean
    X /= X_std
    X[np.isnan(X)] = 0
    return X_mean, X_std

# Multiplication and eigendecomposition
def pca_ref(X, out_pref, dim_ref=DIM_REF, save_XTXsV=False):
    logging.info('Calculating covariance matrix...')
    XTX = X.T @ X
    logging.info('Eigendecomposition on covariance matrix...')
    s, V = svd_eigcov(XTX)
    pcs_ref = V[:, :dim_ref] * s[:dim_ref]
    np.savetxt(out_pref+'_ref.dat', pcs_ref, fmt=NP_OUTPUT_FMT)
    if save_XTXsV:
        np.savetxt(out_pref+'_XTX.dat', XTX, fmt=NP_OUTPUT_FMT)
        np.savetxt(out_pref+'_s.dat', s, fmt=NP_OUTPUT_FMT)
        np.savetxt(out_pref+'_V.dat', V, fmt=NP_OUTPUT_FMT)
    return s, V, pcs_ref

def get_pc_loading(X, s, V, dim_study_high=DIM_STUDY_HIGH, out_pref=None):
    logging.info('Calculating PC loadings...')
    U = X @ (V[:,:dim_study_high] / s[:dim_study_high])
    if out_pref is not None:
        np.savetxt(out_pref+'_U.dat', U, fmt=NP_OUTPUT_FMT)
    return U

def online_procrustes(U, s, V, b, pcs_ref, dim_ref=DIM_REF, dim_study=DIM_STUDY, dim_study_high=DIM_STUDY_HIGH, dim_svdonline=DIM_SVDONLINE):
    s_new, V_new = svd_online(U[:,:dim_study_high], s[:dim_study_high], V[:,:dim_study_high], b, DIM_SVDONLINE)
    s_new, V_new = s_new[:dim_study], V_new[:, :dim_study]
    pcs_new = V_new * s_new
    pcs_new_head, pcs_new_tail = pcs_new[:-1, :], pcs_new[-1, :].reshape((1,-1))
    R, rho, c = procrustes_diffdim(pcs_ref, pcs_new_head)
    pcs_new_tail_trsfed = pcs_new_tail @ R * rho + c
    pcs_stu_onl = pcs_new_tail_trsfed.flatten()
    return pcs_stu_onl

def hdpca_adjust(s, p_ref, n_ref, pcs_stu_proj, hdpca_n_spike_max=HDPCA_N_SPIKE_MAX):
    # Run hdpca but suppress output to stdout
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        pcs_stu_hdpca = np.array(pc_adjust(s**2, p_ref, n_ref, pcs_stu_proj, n_spikes_max=hdpca_n_spike_max))
        sys.stdout = old_stdout
    return pcs_stu_hdpca

def pca_stu(W_large, X_mean, X_std, U, s, V, pcs_ref, out_pref, dim_ref=DIM_REF, dim_study=DIM_STUDY, dim_study_high=DIM_STUDY_HIGH, chunk_size_study=CHUNK_SIZE_STUDY):
    p_ref = len(X_mean)
    n_ref = pcs_ref.shape[0]
    p_stu, n_stu = W_large.shape
    pcs_stu_proj = np.zeros((n_stu, dim_ref), dtype=np.float32)
    pcs_stu_hdpca = np.zeros((n_stu, dim_ref), dtype=np.float32)
    pcs_stu_onl = np.zeros((n_stu, dim_ref), dtype=np.float32)
    chunk_n_stu = int(np.ceil(n_stu / chunk_size_study))
    logging.info('Calculating study PC scores...')
    elapse_subset = 0.0
    elapse_standardize = 0.0
    elapse_proj = 0.0
    elapse_hdpca = 0.0
    elapse_onl = 0.0

    for i in range(chunk_n_stu):
        logging.debug('Subsetting study samples...')
        t0 = time.time()
        sample_start = chunk_size_study * i 
        sample_end = min(chunk_size_study * (i+1), n_stu)
        W = W_large[:, sample_start:sample_end]
        if type(W) is da.core.Array:
            W = W.compute()
        elapse_subset += time.time() - t0

        t0 = time.time()
        logging.debug('Centering, normalizing, and imputing study data...')
        W -= X_mean
        W /= X_std
        W[np.isnan(W)] = 0
        elapse_standardize += time.time() - t0

        t0 = time.time()
        logging.debug('Calculating study pc scores with simple projection...')
        pcs_stu_proj_dim_study = W.T @ U[:,:dim_study]
        pcs_stu_proj[sample_start:sample_end, :] = pcs_stu_proj_dim_study[:, :dim_ref]
        elapse_proj += time.time() - t0

        t0 = time.time()
        logging.debug('Adjusting simple projection pcs with hdpca...')

        pcs_stu_hdpca[sample_start:sample_end, :] = hdpca_adjust(s, p_ref, n_ref, pcs_stu_proj_dim_study)[:, :dim_ref]
        elapse_hdpca += time.time() - t0

        t0 = time.time()
        logging.debug('Calculating study pc scores with svd_online...')
        for i in range(W.shape[1]):
            b = W[:,i]
            pcs_stu_onl[sample_start + i, :] = online_procrustes(U, s, V, b, pcs_ref)[:dim_ref]
        elapse_onl += time.time() - t0
        logging.info('Finished analyzing ' + str(sample_end) + ' samples.')

    logging.info('Finished analyzing study samples.')
    logging.info('Runtimes: ')
    logging.info('Standardizing: ' + str(elapse_standardize))
    logging.info('Subsetting: ' + str(elapse_subset))
    logging.info('Projection: ' + str(elapse_proj))
    logging.info('HDPCA: ' + str(elapse_hdpca))
    logging.info('Online: ' + str(elapse_onl))

    logging.info('Saving study PC scores...')
    np.savetxt(out_pref+'_stu_proj.dat', pcs_stu_proj, fmt=NP_OUTPUT_FMT, delimiter='\t')
    np.savetxt(out_pref+'_stu_hdpca.dat', pcs_stu_hdpca, fmt=NP_OUTPUT_FMT, delimiter='\t')
    np.savetxt(out_pref+'_stu_onl.dat', pcs_stu_onl, fmt=NP_OUTPUT_FMT, delimiter='\t')
    logging.info('Study PC scores saved to ' + out_pref + '_stu_[method].dat')
    del W

    return pcs_stu_proj, pcs_stu_hdpca, pcs_stu_onl

def pred_popu_stu(pcs_ref, popu_ref, pcs_stu):
    logging.info('Predicting populations for study individuals...')
    knn = KNeighborsClassifier(n_neighbors=N_NEIGHBORS)
    knn.fit(pcs_ref, popu_ref)
    popu_stu_pred = knn.predict(pcs_stu) 
    return popu_stu_pred

def load_pcs(pref):
    logging.info('Loading existing reference and study PC scores...')
    pcs_ref = np.loadtxt(pref+'_ref.dat')
    pcs_stu_proj = np.loadtxt(pref+'_stu_proj.dat')
    pcs_stu_hdpca = np.loadtxt(pref+'_stu_hdpca.dat')
    pcs_stu_onl = np.loadtxt(pref+'_stu_onl.dat')
    return pcs_ref, pcs_stu_proj, pcs_stu_hdpca, pcs_stu_onl

def plot_pcs(pcs_ref, pcs_stu_proj, pcs_stu_hdpca, pcs_stu_onl, popu_ref, popu_stu, out_pref, marker_ref='s', marker_stu='.', alpha_ref=PLOT_ALPHA_REF, alpha_stu=PLOT_ALPHA_STU):
    popu_unique = set(popu_ref)
    popu_n = len(popu_unique)
    plot_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    n_subplot = DIM_REF // 2
    fig, ax = plt.subplots(ncols=n_subplot)
    for j in range(n_subplot):
        plt.subplot(1, n_subplot, j+1)
        plt.xlabel('PC' + str(j*2+1))
        plt.ylabel('PC' + str(j*2+2))
        for i,popu in enumerate(popu_unique):
            ref_is_this_popu = popu_ref == popu
            plt.plot(pcs_ref[ref_is_this_popu, j*2], pcs_ref[ref_is_this_popu, j*2+1], marker_ref, alpha=alpha_ref, color=plot_colors[i])
        for i,popu in enumerate(popu_unique):
            stu_is_this_popu = popu_stu == popu
            plt.plot(pcs_stu_onl[stu_is_this_popu, j*2], pcs_stu_onl[stu_is_this_popu, j*2+1], marker_stu, label=str(popu), alpha=alpha_stu, color=plot_colors[i])
        # plt.plot(pcs_stu_proj[:, j*2], pcs_stu_proj[:, j*2+1], '+', alpha=alpha_stu, label='projection', color=PLOT_COLOR_STU)
        # plt.plot(pcs_stu_hdpca[:, j*2], pcs_stu_hdpca[:, j*2+1], 'x', alpha=alpha_stu, label='hdpca', color=PLOT_COLOR_STU)
        # plt.plot(pcs_stu_trace[:, j*2], pcs_stu_trace[:, j*2+1], 'o', alpha=alpha_stu, label='trace')
    # plt.legend(ncol=2, loc='center left', bbox_to_anchor=(1, 0.5))
    # plt.legend(loc='lower center', bbox_to_anchor=(0.5, 0), bbox_transform=fig.transFigure, fancybox=True, shadow=True, ncol=popu_n*2)
    plt.legend()
    # fig.subplots_adjust(bottom=0.17)
    plt.tight_layout()
    plt.savefig(out_pref+'.png', dpi=300)
    plt.close('all')
    logging.info('Study PC score plots saved to ' + out_pref +'.png')

def pca(X, W_dask, out_pref):
    # PCA on ref and stu
    X_mean, X_std = standardize_ref(X)
    s, V, pcs_ref = pca_ref(X, out_pref)
    U = get_pc_loading(X, s, V)
    pcs_stu_proj, pcs_stu_hdpca, pcs_stu_onl = pca_stu(W_dask, X_mean, X_std, U, s, V, pcs_ref, out_pref)
    return pcs_ref, pcs_stu_proj, pcs_stu_hdpca, pcs_stu_onl


def run_pca(ref_pref, stu_pref, popu_ref_filename=None, popu_ref_k=None, use_memmap=False):
    assert (popu_ref_filename is None) != (popu_ref_k is None)
    logging.info('Reference data: ' + ref_pref)
    logging.info('Study data: ' + stu_pref)
    logging.info('Temp dir: ' + TMP_DIR)
    # Intersect ref and stu snps
    ref_pref_commsnpsrefal, stu_pref_commsnpsrefal = intersect_ref_stu_snps(ref_pref, stu_pref)
    # Load ref
    X_dask, X_bim, X_fam = bed2dask(ref_pref_commsnpsrefal)
    if use_memmap:
        X = dask2memmap(X_dask, 'X.memmap')
    else:
        X = X_dask.compute()
    # popu_ref = get_popu_ref_info(X_fam, popu_ref_filename, superpopu_ref_filename)[popu_col_name]
    # Load stu
    W_dask, W_bim, W_fam = bed2dask(stu_pref_commsnpsrefal)
    # Check ref and stu have the same snps and alleles
    assert X_bim.shape == W_bim.shape
    assert all(X_bim[['chrom', 'snp', 'pos', 'a0', 'a1']] == W_bim[['chrom', 'snp', 'pos', 'a0', 'a1']])
    p_ref, n_ref = X.shape
    p_stu, n_stu = W_dask.shape
    assert p_ref == p_stu
    p = p_ref

    # PCA on study individuals
    pcs_ref, pcs_stu_proj, pcs_stu_hdpca, pcs_stu_onl = pca(X, W_dask, stu_pref)

    # Read or predict ref populations
    if popu_ref_filename is not None:
        popu_ref = pd.read_table(popu_ref_filename, header=None).iloc[:,1]
    else:
        popu_ref = KMeans(n_clusters=popu_ref_k).fit_predict(pcs_ref)
        popu_ref_df = pd.DataFrame({'iid':X_fam['iid'], 'popu':popu_ref})
        popu_ref_df.to_csv(stu_pref+'_ref_pred.popu', sep=DELIMITER, header=False, index=False)

    # Predict stu population
    popu_stu_pred = pred_popu_stu(pcs_ref, popu_ref, pcs_stu_onl)

    # Plot PC scores
    plot_pcs(pcs_ref, pcs_stu_proj, pcs_stu_hdpca, pcs_stu_onl, popu_ref, popu_stu_pred, stu_pref)

    # Finer-level PCA on European individuals
    # ref_indiv_is_eur = popu_ref == 'EUR'
    # stu_indiv_is_eur = popu_stu_pred == 'EUR'
    # X_fam_eur = X_fam[ref_indiv_is_eur]
    # W_fam_eur = W_fam[stu_indiv_is_eur]
    # X_fam_eur.to_csv(ref_pref+'_eur.id', sep=DELIMITER, header=False, columns=['fid', 'iid'], index=False)
    # W_fam_eur.to_csv(stu_pref+'_eur.id', sep=DELIMITER, header=False, columns=['fid', 'iid'], index=False)
    # X_eur = X[:, ref_indiv_is_eur]
    # popu_ref_eur = popu_ref[ref_indiv_is_eur]
    # # The following line actually slows down pca on study samples quite a lot, especially hdpca
    # # Actually not necessarily. Could be that eur indivs are harder for hdpca to adjust.
    # W_dask_eur = W_dask[:, stu_indiv_is_eur]
    # # W_eur = dask2memmap(W_dask_eur, 'W_eur.memmap')
    # pca(X_eur, W_dask_eur, popu_ref_eur, 'superpopu_eur')
    # # pcs_ref_eur = pcs_ref[indiv_is_eur,:] * X_std[indiv_is_eur]
    # # pcs_ref_eur += X_mean[indiv_is_eur]

    del X
    # logging.info('Temporary directory content: ')
    # logging.info(subprocess.run(['ls', '-hl', TMP_DIR]))

