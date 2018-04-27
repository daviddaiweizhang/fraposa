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
hdpc_est = r['hdpc_est']
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
import filecmp

DIM_REF = 4
DIM_STU = 20
DIM_STU_HIGH = DIM_STU * 2
N_NEIGHBORS=5
HDPCA_N_SPIKES_MAX = 20
CHUNK_SIZE_STUDY = 5000

PLOT_ALPHA_REF=0.1
PLOT_ALPHA_STU=0.99
PLOT_MARKERS = ['.', '+', 'x', '*', 'd', 's']

# DIM_SVDRAND = DIM_STU * 4
# NITER_SVDRAND = 2
NP_OUTPUT_FMT = '%.4f'
DELIMITER = '\t'
TMP_DIR = mkdtemp()

# def hdpca_adjust(s, p_ref, n_ref, pcs_stu_proj, hdpca_n_spikes_max=HDPCA_N_SPIKES_MAX):
#     # Run hdpca but suppress output to stdout
#     with open(os.devnull, 'w') as devnull:
#         old_stdout = sys.stdout
#         sys.stdout = devnull
#         pcs_stu_hdpca = np.array(pc_adjust(s**2, p_ref, n_ref, pcs_stu_proj, n_spikes_max=hdpca_n_spikes_max))
#         sys.stdout = old_stdout
#     return pcs_stu_hdpca

def create_logger(prefix=None, level='info'):
    log = logging.getLogger()
    if level == 'info':
        log_level = logging.INFO
    elif level == 'debug':
        log_level = logging.DEBUG
    else:
        assert False
    log.handlers = [] # Avoid duplicated logs in interactive modes
    log.setLevel(log_level)
    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    # create file handler which logs even debug messages
    filename = prefix + '.' + str(round(time.time())) + '.log'
    fh = logging.FileHandler(filename, 'w')
    fh.setLevel(log_level)
    fh.setFormatter(formatter)
    log.addHandler(fh)
    # create console handler with a higher log level
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(log_level)
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

def svd_online(U1, d1, V1, b, l=None):
    n, k = V1.shape
    if l is None:
        l = k
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
    logging.debug('Testing test_svd_online...')

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
    logging.debug('Passed!')

    logging.debug('Testing procrustes...')
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
    logging.debug('Passed!')

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
    snps_are_identical = filecmp.cmp(ref_pref+'.bim', stu_pref+'.bim')
    if snps_are_identical:
        logging.info('SNPs and alleles in reference and study samples are identical')
        return ref_pref, stu_pref
    else:
        logging.info('Intersecting SNPs in reference and study samples...')
        bashout = subprocess.run(['bash', 'intersect_bed.sh', ref_pref, stu_pref, TMP_DIR], stdout=subprocess.PIPE)
        ref_pref_commsnpsrefal, stu_pref_commsnpsrefal = bashout.stdout.decode('utf-8').split('\n')[-3:-1]
        assert len(ref_pref_commsnpsrefal) > 0
        assert len(stu_pref_commsnpsrefal) > 0
        return ref_pref_commsnpsrefal, stu_pref_commsnpsrefal

def bed2dask(filename, X_dtype=np.float32):
    logging.debug('Reading Plink binary data into dask array...')
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
    logging.debug('Centering, normalizing, and imputing reference data...')
    X_mean = np.nanmean(X, axis = 1).reshape((-1, 1))
    X_std = np.nanstd(X, axis = 1).reshape((-1,1))
    X_std[X_std == 0] = 1
    X -= X_mean
    X /= X_std
    X[np.isnan(X)] = 0
    return X_mean, X_std

# Multiplication and eigendecomposition
def pca_ref(X, dim_ref=DIM_REF, save_XTXsV=False):
    logging.info('Calculating covariance matrix...')
    XTX = X.T @ X
    logging.info('Eigendecomposition on covariance matrix...')
    s, V = svd_eigcov(XTX)
    # if save_XTXsV:
    #     np.savetxt(out_pref+'_XTX.dat', XTX, fmt=NP_OUTPUT_FMT)
    #     np.savetxt(out_pref+'_s.dat', s, fmt=NP_OUTPUT_FMT)
    #     np.savetxt(out_pref+'_V.dat', V, fmt=NP_OUTPUT_FMT)
    return s, V, XTX

def get_pc_loading(X, s, V, dim_stu_high=DIM_STU_HIGH, out_pref=None):
    logging.info('Calculating PC loadings...')
    U = X @ (V[:,:dim_stu_high] / s[:dim_stu_high])
    if out_pref is not None:
        np.savetxt(out_pref+'_U.dat', U, fmt=NP_OUTPUT_FMT)
    return U

def ref_aug_procrustes(pcs_ref, pcs_aug):
    n_ref, p_ref = pcs_ref.shape
    n_aug, p_aug = pcs_aug.shape
    assert n_aug == n_ref + 1
    assert p_aug >= p_ref
    pcs_aug_head = pcs_aug[:-1, :] 
    pcs_aug_tail = pcs_aug[-1, :].reshape((1,-1))
    R, rho, c = procrustes_diffdim(pcs_ref, pcs_aug_head)
    pcs_aug_tail_trsfed = pcs_aug_tail @ R * rho + c
    return pcs_aug_tail_trsfed.flatten()

def oadp(U, s, V, b, dim_ref=DIM_REF, dim_stu=DIM_STU, dim_stu_high=DIM_STU_HIGH):
    pcs_ref = V[:, :dim_ref] * s[:dim_ref]
    s_aug, V_aug = svd_online(U[:,:dim_stu_high], s[:dim_stu_high], V[:,:dim_stu_high], b)
    s_aug, V_aug = s_aug[:dim_stu], V_aug[:, :dim_stu]
    pcs_aug = V_aug * s_aug
    pcs_stu = ref_aug_procrustes(pcs_ref, pcs_aug)
    return pcs_stu[:dim_ref]

def adp(XTX, X, w, pcs_ref, dim_stu=DIM_STU):
    dim_ref = pcs_ref.shape[1]
    w = w.reshape((-1,1))
    XTw = X.T @ w
    wTw = w.T @ w
    XTX_aug = np.vstack((np.hstack((XTX, XTw)), np.hstack((XTw.T, wTw))))
    s_aug, V_aug = svd_eigcov(XTX_aug)
    s_aug = s_aug[:dim_stu] 
    V_aug = V_aug[:, :dim_stu]
    pcs_aug = V_aug * s_aug
    pcs_stu = ref_aug_procrustes(pcs_ref, pcs_aug)
    return pcs_stu[:dim_ref]

def pca_stu(W_large, X_mean, X_std, method='oadp',
            U=None, s=None, V=None, XTX=None, X=None, pcs_ref=None,
            dim_ref=DIM_REF, dim_stu=DIM_STU, dim_stu_high=DIM_STU_HIGH):
    logging.info('Calculating study PC scores with ' + method + '...')
    p_ref = len(X_mean)
    n_ref = len(s)
    p_stu, n_stu = W_large.shape
    chunk_n_stu = int(np.ceil(n_stu / CHUNK_SIZE_STUDY))
    pcs_stu = np.zeros((n_stu, dim_ref), dtype=np.float32)
    elapse_subset = 0.0
    elapse_standardize = 0.0
    elapse_method = 0.0

    for i in range(chunk_n_stu):
        logging.debug('Subsetting study samples...')
        t0 = time.time()
        sample_start = CHUNK_SIZE_STUDY * i 
        sample_end = min(CHUNK_SIZE_STUDY * (i+1), n_stu)
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
        if method == 'oadp':
            logging.debug('Predicting study pc scores with online augment-decompose-procrustes...')
            assert (U is not None) & (s is not None) and (V is not None)
            for i in range(W.shape[1]):
                w = W[:,i]
                pcs_stu_row = oadp(U, s, V, w, dim_ref, dim_stu, dim_stu_high)
                pcs_stu[sample_start + i, :] = pcs_stu_row
        elif method == 'adp':
            logging.debug('Predicting study pc scores with augment-decompose-procrustes...')
            assert (XTX is not None) and (X is not None)
            for i in range(W.shape[1]):
                w = W[:,i]
                pcs_stu_row = adp(XTX, X, w, pcs_ref, dim_stu=dim_stu)
                pcs_stu[sample_start + i, :] = pcs_stu_row
        else:
            assert U is not None
            pcs_stu_chunk = W.T @ (U[:,:dim_ref])
            pcs_stu[sample_start:sample_end, :] = pcs_stu_chunk
            if method == 'sp':
                logging.debug('Predicting study PC scores with simple projection...')
            elif method == 'ap':
                logging.debug('Predicting study PC scores with adjusted projection...')
            else:
                logging.error(method + ' is not one of sp, ap, adp, or oadp.')
        elapse_method += time.time() - t0

        logging.info('Finished analyzing ' + str(sample_end) + ' samples.')

    logging.info('Finished analyzing all study samples.')
    logging.info('Runtimes: ')
    logging.info('Standardizing: ' + str(elapse_standardize))
    logging.info('Subsetting: ' + str(elapse_subset))
    logging.info(method + ': ' + str(elapse_method))
    del W
    return pcs_stu

def pred_popu_stu(pcs_ref, popu_ref, pcs_stu):
    logging.info('Predicting populations for study individuals...')
    knn = KNeighborsClassifier(n_neighbors=N_NEIGHBORS)
    knn.fit(pcs_ref, popu_ref)
    popu_stu_pred = knn.predict(pcs_stu) 
    return popu_stu_pred

def load_pcs(pref, methods):
    logging.info('Loading existing reference and study PC scores...')
    pcs_ref = np.loadtxt(pref+'_ref.pcs')
    pcs_stu_list = []
    for mth in methods:
        pcs_stu_filename = pref + '_stu_' + mth + '.pcs'
        pcs_stu_this = np.loadtxt(pcs_stu_filename)
        pcs_stu_list += [pcs_stu_this]
    return pcs_ref, pcs_stu_list


def plot_pcs(pcs_ref, pcs_stu_list, popu_ref, popu_stu_list, method_list, out_pref,
             markers=PLOT_MARKERS, alpha_ref=PLOT_ALPHA_REF, alpha_stu=PLOT_ALPHA_STU):
    assert len(pcs_stu_list) == len(popu_stu_list) == len(method_list)
    popu_unique = set(popu_ref)
    popu_n = len(popu_unique)
    plot_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    dim_ref = pcs_ref.shape[1]
    n_subplot = dim_ref // 2
    fig, ax = plt.subplots(ncols=n_subplot)
    for j in range(n_subplot):
        plt.subplot(1, n_subplot, j+1)
        plt.xlabel('PC' + str(j*2+1))
        plt.ylabel('PC' + str(j*2+2))
        for i,popu in enumerate(popu_unique):
            ref_is_this_popu = popu_ref == popu
            plt.plot(pcs_ref[ref_is_this_popu, j*2], pcs_ref[ref_is_this_popu, j*2+1], markers[-1], alpha=alpha_ref, color=plot_colors[i], label=str(popu))
        if len(pcs_stu_list) > 0:
            for k,pcs_stu in enumerate(pcs_stu_list):
                popu_stu = popu_stu_list[k]
                method = method_list[k]
                for i,popu in enumerate(popu_unique):
                    stu_is_this_popu = popu_stu == popu
                    if np.sum(stu_is_this_popu) > 0:
                        plot_label = None
                        if i == 0:
                            plot_label = str(method)
                        plt.plot(pcs_stu[stu_is_this_popu, j*2], pcs_stu[stu_is_this_popu, j*2+1], markers[k], label=plot_label, alpha=alpha_stu, color=plot_colors[i])
    plt.legend()
    plt.tight_layout()
    fig_filename = out_pref+'_'.join([''] + method_list)+'.png'
    plt.savefig(fig_filename, dpi=300)
    plt.close('all')
    logging.info('PC plots saved to ' + fig_filename)

def adj_hdpc_shrinkage(U, s, p_ref, n_ref, hdpca_n_spikes_max=HDPCA_N_SPIKES_MAX, dim_ref=DIM_REF):
    logging.info('Estimating PC shrinkage and adjusting PC loadings...')
    hdpc_est_result  = hdpc_est(s**2, p_ref, n_ref, n_spikes_max=hdpca_n_spikes_max)
    shrinkage = np.array(hdpc_est_result[-1])
    n_pc_adjusted = min(dim_ref, len(shrinkage))
    for i in range(n_pc_adjusted):
        U[:, i] /= shrinkage[i]

def pca(X, W_dask, out_pref, method='oadp',
        dim_ref=DIM_REF, dim_stu=DIM_STU, dim_stu_high=DIM_STU_HIGH, hdpca_n_spikes_max=HDPCA_N_SPIKES_MAX):
    # PCA on ref and stu
    p_ref, n_ref = X.shape
    X_mean, X_std = standardize_ref(X)
    s, V, XTX = pca_ref(X)
    pcs_ref = V[:, :dim_ref] * s[:dim_ref]
    np.savetxt(out_pref+'_ref.pcs', pcs_ref, fmt=NP_OUTPUT_FMT)
    logging.info('Reference PC scores saved to ' + out_pref + '_ref.pcs')
    U = get_pc_loading(X, s, V)

    if method == 'ap':
        adj_hdpc_shrinkage(U, s, p_ref, n_ref, dim_ref)

    pcs_stu = pca_stu(W_dask, X_mean, X_std, method=method,
                      U=U, s=s, V=V, XTX=XTX, X=X, pcs_ref=pcs_ref,
                      dim_ref=dim_ref, dim_stu=dim_stu, dim_stu_high=dim_stu_high)
    pcs_stu_filename = out_pref + '_stu_' + method +'.pcs'
    np.savetxt(pcs_stu_filename, pcs_stu, fmt=NP_OUTPUT_FMT, delimiter='\t')
    logging.info('Study PC scores saved to ' + pcs_stu_filename)
    return pcs_ref, pcs_stu

def run_pca(ref_pref, stu_pref, popu_ref_filename=None, popu_ref_k=None, method='oadp', dim_ref=DIM_REF, dim_stu=DIM_STU, dim_stu_high=DIM_STU_HIGH, use_memmap=False):
    assert 2 <= dim_ref <= dim_stu <= dim_stu_high
    log = create_logger(stu_pref)
    assert (popu_ref_filename is None) != (popu_ref_k is None)
    logging.info('Reference data: ' + ref_pref)
    logging.info('Study data: ' + stu_pref)
    logging.debug('Temp dir: ' + TMP_DIR)
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
    pcs_ref, pcs_stu = pca(X, W_dask, stu_pref, method, dim_ref, dim_stu, dim_stu_high)

    # Read or predict ref populations
    if popu_ref_filename is not None:
        logging.info('Reading reference population from ' + popu_ref_filename)
        popu_ref = pd.read_table(popu_ref_filename, header=None).iloc[:,1]
    else:
        logging.info('Predicting reference population...')
        popu_ref = KMeans(n_clusters=popu_ref_k).fit_predict(pcs_ref)
        popu_ref_df = pd.DataFrame({'iid':X_fam['iid'], 'popu':popu_ref})
        popu_ref_df.to_csv(ref_pref+'_pred.popu', sep=DELIMITER, header=False, index=False)
        logging.info('Reference population prediction saved to ' + ref_pref+'_pred.popu')

    # Predict stu population
    popu_stu_pred = pred_popu_stu(pcs_ref, popu_ref, pcs_stu)
    popu_stu_pred_df = pd.DataFrame({'iid':W_fam['iid'], 'popu':popu_stu_pred})
    popu_stu_pred_df.to_csv(stu_pref+'_pred.popu', sep=DELIMITER, header=False, index=False)
    logging.info('Study population prediction saved to ' + stu_pref+'_pred.popu')

    # Plot PC scores
    plot_pcs(pcs_ref, [pcs_stu], popu_ref, [popu_stu_pred], method_list=[method], out_pref=stu_pref)

    # logging.info('Temporary directory content: ')
    # logging.info(subprocess.run(['ls', '-hl', TMP_DIR]))

    # Finer-level PCA on a subpopulation
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
    return pcs_ref, pcs_stu, popu_ref, popu_stu_pred
