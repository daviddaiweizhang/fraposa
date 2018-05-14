import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
# from pandas_plink import read_plink
from pyplink import PyPlink
# import dask.array as da
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
HDPCA_N_SPIKES_MAX = 18
SAMPLE_CHUNK_SIZE_STU = 50
SAMPLE_SPLIT_PREF_LEN = 4
PROCRUSTES_NITER_MAX = 10000
PROCRUSTES_EPSILON_MIN = 1e-6

PLOT_ALPHA_REF=0.1
PLOT_ALPHA_STU=0.99
PLOT_MARKERS = ['.', '+', 'x', '*', 'd', 's']

# DIM_SVDRAND = DIM_STU * 4
# NITER_SVDRAND = 2
NP_OUTPUT_FMT = '%.4f'
DELIMITER = '\t'
# DIR_TMP = mkdtemp()
DIR_TMP = 'tmp'

def create_logger(prefix='frugalpca', level='info'):
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

def procrustes(Y_mat, X_mat, return_transformed=False):
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
    trS = np.sum(s)
    R = VT.T @ U.T
    rho = trS / trXX
    c = Y_mean - rho * X_mean @ R
    if return_transformed:
        X_new = X_mat @ R * rho + c
        return R, rho, c, X_new
    else:
        return R, rho, c

def procrustes_similarity(Y_mat, X_mat):
    X = np.array(X_mat, dtype=np.double, copy=True)
    Y = np.array(Y_mat, dtype=np.double, copy=True)
    assert X.shape == Y.shape
    X_transformed = procrustes(Y, X, return_transformed=True)[-1]
    Z = Y - X_transformed
    trZZ = np.sum(Z**2)
    Y_mean = np.mean(Y, 0)
    Y -= Y_mean
    trYY = np.sum(Y**2)
    similarity = 1 - (trZZ / trYY)
    assert 0 <= similarity <= 1
    return similarity

def procrustes_diffdim(Y_mat, X_mat, n_iter_max=PROCRUSTES_NITER_MAX, epsilon_min=PROCRUSTES_EPSILON_MIN, return_transformed=False):
    X = np.array(X_mat, dtype=np.double, copy=True)
    Y = np.array(Y_mat, dtype=np.double, copy=True)
    n_X, p_X = X.shape
    n_Y, p_Y = Y.shape
    assert n_X == n_Y
    assert p_X >= p_Y
    if p_X == p_Y:
        return procrustes(Y, X, return_transformed)
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
        if return_transformed:
            return R, rho, c, X_new
        else:
            return R, rho, c

def load_trace(filename, isref=False):
    trace_df = pd.read_table(filename)
    if isref:
        n_col_skip = 2
    else:
        n_col_skip = 6
    trace_pcs = trace_df.iloc[:, n_col_skip:].values
    return trace_pcs

def intersect_ref_stu_snps(pref_ref, pref_stu, path_tmp):
    snps_are_identical = filecmp.cmp(pref_ref+'.bim', pref_stu+'.bim')
    if snps_are_identical:
        logging.info('SNPs and alleles in reference and study samples are identical')
        return pref_ref, pref_stu
    else:
        logging.info('Intersecting SNPs in reference and study samples...')
        bashout = subprocess.run(['bash', 'intersect_bed.sh', pref_ref, pref_stu, path_tmp], stdout=subprocess.PIPE)
        pref_ref_commsnpsrefal, pref_stu_commsnpsrefal = bashout.stdout.decode('utf-8').split('\n')[-3:-1]
        assert len(pref_ref_commsnpsrefal) > 0
        assert len(pref_stu_commsnpsrefal) > 0
        return pref_ref_commsnpsrefal, pref_stu_commsnpsrefal

def bed2dask(filename, dtype=np.float32):
    logging.debug('Reading Plink binary data into dask array...')
    X_bim, X_fam, X_dask = read_plink(filename, verbose=False)
    X_dask = X_dask.astype(dtype)
    return X_dask, X_bim, X_fam

def read_bed(bed_filepref, out_type='memory', dtype=np.int8):
    logging.debug('Reading Plink binary data into Numpy array...')
    ped = PyPlink(bed_filepref)
    bim = ped.get_bim()
    fam = ped.get_fam()
    if out_type == 'ped':
        return ped, bim, fam
    else:
        p = ped.get_nb_markers()
        n = ped.get_nb_samples()
        if out_type in ['memmap_C', 'memmap_F']:
            memmap_filename = bed_filepref + '.memmap'
            order = out_type[-1]
            mat = np.memmap(memmap_filename, dtype=dtype, mode='w+', shape=(p, n), order=order)
        elif out_type == 'memory':
            mat = np.zeros(shape=(p,n), dtype=dtype)
        else:
            logging.error('out_type ' + str(out_type) + ' is not in {memory, memmap, ped}')
            assert False
        for (i, (marker, geno)) in enumerate(ped):
            mat[i,:] = geno
        return mat, bim, fam

def dask2memmap(X_dask, X_memmap_filename, path_tmp):
    logging.info('Loading dask array into memmap...')
    dtype = X_dask.dtype
    X_memmap_filename = os.path.join(path_tmp, X_memmap_filename)
    X = np.memmap(X_memmap_filename, dtype=dtype, mode='w+', shape=X_dask.shape)
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

def pca_stu(stu_bed_filepref, X_mean, X_std, method, path_tmp,
            U=None, s=None, V=None, XTX=None, X=None, pcs_ref=None,
            dim_ref=DIM_REF, dim_stu=DIM_STU, dim_stu_high=DIM_STU_HIGH):
    p_ref = len(X_mean)
    n_ref = len(s)
    ped = PyPlink(stu_bed_filepref)
    p_stu = ped.get_nb_markers()
    n_stu = ped.get_nb_samples()
    chunk_n_stu = int(np.ceil(n_stu / SAMPLE_CHUNK_SIZE_STU))
    pcs_stu = np.zeros((n_stu, dim_ref))

    logging.info('Spliting study samples...')
    t0 = time.time()
    bashout = subprocess.run(['bash', 'split_fam.sh', stu_bed_filepref, str(SAMPLE_CHUNK_SIZE_STU)], stdout=subprocess.PIPE)
    stu_bed_filepref_chunk_list = bashout.stdout.decode('utf-8').split('\n')[-2].split()
    assert len(stu_bed_filepref_chunk_list) == chunk_n_stu
    elapse_split = time.time() - t0

    elapse_load = 0.0
    elapse_standardize = 0.0
    elapse_method = 0.0

    logging.info('Calculating study PC scores with ' + method + '...')
    for i in range(chunk_n_stu):
        logging.debug('Reading study samples...')
        t0 = time.time()
        sample_start = SAMPLE_CHUNK_SIZE_STU * i
        sample_end = min(SAMPLE_CHUNK_SIZE_STU * (i+1), n_stu)
        W = read_bed(stu_bed_filepref_chunk_list[i], out_type='memory', dtype=np.float32)[0]
        elapse_load += time.time() - t0

        t0 = time.time()
        logging.debug('Centering, normalizing, and imputing study data...')
        logging.debug('Centering...')
        W -= X_mean
        logging.debug('Normalizing...')
        W /= X_std
        logging.debug('Imputing...')
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
    logging.info('Splitting: ' + str(elapse_split))
    logging.info('Loading: ' + str(elapse_load))
    logging.info('Standardizing: ' + str(elapse_standardize))
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

def pca(X, W_filepref, out_pref, method, path_tmp,
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

    pcs_stu = pca_stu(W_filepref, X_mean, X_std, method=method, path_tmp=path_tmp,
                      U=U, s=s, V=V, XTX=XTX, X=X, pcs_ref=pcs_ref,
                      dim_ref=dim_ref, dim_stu=dim_stu, dim_stu_high=dim_stu_high)
    pcs_stu_filename = out_pref + '_stu_' + method +'.pcs'
    np.savetxt(pcs_stu_filename, pcs_stu, fmt=NP_OUTPUT_FMT, delimiter='\t')
    logging.info('Study PC scores saved to ' + pcs_stu_filename)
    return pcs_ref, pcs_stu

def run_pca(pref_ref, pref_stu, popu_ref_filename=None, popu_ref_k=None, method='oadp', dim_ref=DIM_REF, dim_stu=DIM_STU, dim_stu_high=DIM_STU_HIGH, use_memmap=False, log_level='info'):
    dir_ref = os.path.dirname(pref_ref)
    dir_stu = os.path.dirname(pref_stu)
    path_tmp = os.path.join(dir_stu, DIR_TMP)
    assert 2 <= dim_ref <= dim_stu <= dim_stu_high
    log = create_logger(pref_stu, log_level)
    assert (popu_ref_filename is None) != (popu_ref_k is None)
    logging.info('Reference data: ' + pref_ref)
    logging.info('Study data: ' + pref_stu)
    logging.debug('Tmp path: ' + path_tmp)
    subprocess.run(['mkdir', '-p', path_tmp])
    # Intersect ref and stu snps
    pref_ref_commsnpsrefal, pref_stu_commsnpsrefal = intersect_ref_stu_snps(pref_ref, pref_stu, path_tmp)

    # Load ref
    logging.info('Reading reference samples...')
    if use_memmap:
        mem_out_type = 'memmap_C'
    else:
        mem_out_type = 'memory'
    X, X_bim, X_fam = read_bed(pref_ref_commsnpsrefal, out_type=mem_out_type, dtype=np.float32)
    p_ref, n_ref = X.shape

    W_ped, W_bim, W_fam = read_bed(pref_stu_commsnpsrefal, out_type='ped')
    p_stu = W_ped.get_nb_markers()
    n_stu = W_ped.get_nb_samples()
    assert p_ref == p_stu
    p = p_ref

    # Check ref and stu have the same snps and alleles
    assert all(W_bim.index == X_bim.index)
    assert all(X_bim[['chrom', 'pos', 'a1', 'a2']] == W_bim[['chrom', 'pos', 'a1', 'a2']])

    # PCA on study individuals
    pcs_ref, pcs_stu = pca(X, pref_stu_commsnpsrefal, pref_stu, method, path_tmp, dim_ref, dim_stu, dim_stu_high)

    # Read or predict ref populations
    if popu_ref_filename is not None:
        logging.info('Reading reference population from ' + popu_ref_filename)
        popu_ref = pd.read_table(popu_ref_filename, header=None).iloc[:,1]
    else:
        logging.info('Predicting reference population...')
        popu_ref = KMeans(n_clusters=popu_ref_k).fit_predict(pcs_ref)
        popu_ref_df = pd.DataFrame({'iid':X_fam['iid'], 'popu':popu_ref})
        popu_ref_df.to_csv(pref_ref+'_pred.popu', sep=DELIMITER, header=False, index=False)
        logging.info('Reference population prediction saved to ' + pref_ref+'_pred.popu')

    # Predict stu population
    popu_stu_pred = pred_popu_stu(pcs_ref, popu_ref, pcs_stu)
    popu_stu_pred_df = pd.DataFrame({'iid':W_fam['iid'], 'popu':popu_stu_pred})
    popu_stu_pred_df.to_csv(pref_stu+'_pred.popu', sep=DELIMITER, header=False, index=False)
    logging.info('Study population prediction saved to ' + pref_stu+'_pred.popu')

    # Plot PC scores
    plot_pcs(pcs_ref, [pcs_stu], popu_ref, [popu_stu_pred], method_list=[method], out_pref=pref_stu)

    # logging.info('Temporary directory content: ')
    # logging.info(subprocess.run(['ls', '-hl', DIR_TMP]))

    # Finer-level PCA on a subpopulation
    # ref_indiv_is_eur = popu_ref == 'EUR'
    # stu_indiv_is_eur = popu_stu_pred == 'EUR'
    # X_fam_eur = X_fam[ref_indiv_is_eur]
    # W_fam_eur = W_fam[stu_indiv_is_eur]
    # X_fam_eur.to_csv(pref_ref+'_eur.id', sep=DELIMITER, header=False, columns=['fid', 'iid'], index=False)
    # W_fam_eur.to_csv(pref_stu+'_eur.id', sep=DELIMITER, header=False, columns=['fid', 'iid'], index=False)
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
