import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from rpy2.robjects import numpy2ri
r = robjects.r
robjects.numpy2ri.activate()
importr('hdpca')
hdpc_est = r['hdpc_est']
from pyplink import PyPlink
# from pandas_plink import read_plink
# import dask.array as da
# from dask import compute
# from chest import Chest
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
SAMPLE_CHUNK_SIZE_STU = 10000
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
        bashout = subprocess.run(
            ['bash', 'intersect_bed.sh', pref_ref, pref_stu, path_tmp],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        assert len(bashout.stderr.decode('utf-8')) == 0
        pref_ref_commsnpsrefal, pref_stu_commsnpsrefal = bashout.stdout.decode('utf-8').split('\n')[-3:-1]
        assert len(pref_ref_commsnpsrefal) > 0
        assert len(pref_stu_commsnpsrefal) > 0
        return pref_ref_commsnpsrefal, pref_stu_commsnpsrefal

def read_bed(bed_filepref, bed_store='memory', dtype=np.int8):
    pyp = PyPlink(bed_filepref)
    bim = pyp.get_bim()
    fam = pyp.get_fam()
    if bed_store is None:
        bed = None
    else:
        p = len(bim)
        n = len(fam)
        if bed_store in ['memmap_C', 'memmap_F']:
            memmap_filename = bed_filepref + '.memmap'
            order = bed_store[-1]
            bed = np.memmap(memmap_filename, dtype=dtype, mode='w+', shape=(p, n), order=order)
        elif bed_store == 'memory':
            bed = np.zeros(shape=(p, n), dtype=dtype)
        else:
            logging.error('Invalid output type')
            assert False
        for (i, (snp, genotypes)) in enumerate(pyp):
            bed[i,:] = genotypes
        bed = 2 - bed
    return bed, bim, fam

def dask2memmap(X_dask, X_memmap_filename, path_tmp):
    logging.info('Loading dask array into memmap...')
    dtype = X_dask.dtype
    X_memmap_filename = os.path.join(path_tmp, X_memmap_filename)
    X = np.memmap(X_memmap_filename, dtype=dtype, mode='w+', shape=X_dask.shape)
    X[:] = X_dask
    return X

def standardize(X, mean=None, std=None, miss=3):
    assert np.issubdtype(X.dtype, np.floating)
    p, n = X.shape
    is_miss = X == miss
    if (mean is None) or (std is None):
        mean = np.zeros(p)
        std = np.zeros(p)
        for i in range(p):
            row_nomiss = X[i,:][~is_miss[i,:]]
            mean[i] = np.mean(row_nomiss)
            std[i] = np.std(row_nomiss)
        std[std == 0] = 1
    mean = mean.reshape((-1, 1))
    std = std.reshape((-1, 1))
    X -= mean
    X /= std
    X[is_miss] = 0
    return mean, std

def standardize_ref(X):
    logging.debug('Centering, normalizing, and imputing reference data...')
    X_mean = np.nanmean(X, axis = 1).reshape((-1, 1))
    X_std = np.nanstd(X, axis = 1).reshape((-1,1))
    X_std[X_std == 0] = 1
    X -= X_mean
    X /= X_std
    X[np.isnan(X)] = 0
    return X_mean, X_std

def pca_ref(X):
    logging.info('Calculating covariance matrix...')
    XTX = X.T @ X
    logging.info('Eigendecomposition on covariance matrix...')
    s, V = svd_eigcov(XTX)
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
    W_bim, W_fam = read_bed(stu_bed_filepref, bed_store=None)[1:3]
    p_stu = len(W_bim)
    n_stu = len(W_fam)
    chunk_n_stu = int(np.ceil(n_stu / SAMPLE_CHUNK_SIZE_STU))
    pcs_stu = np.zeros((n_stu, dim_ref))

    print('='*80)
    logging.info('Splitting study samples...')
    t0 = time.time()
    bashout = subprocess.run(['bash', 'split_fam.sh', stu_bed_filepref, str(SAMPLE_CHUNK_SIZE_STU)], stdout=subprocess.PIPE)
    stu_bed_filepref_chunk_list = bashout.stdout.decode('utf-8').split('\n')[-2].split()
    assert len(stu_bed_filepref_chunk_list) == chunk_n_stu
    elapse_split = time.time() - t0

    logging.info('Calculating study PC scores with ' + method + '...')
    elapse_load = 0.0
    elapse_standardize = 0.0
    elapse_method = 0.0
    for i in range(chunk_n_stu):
        t0 = time.time()
        logging.debug('Reading study samples...')
        sample_start = SAMPLE_CHUNK_SIZE_STU * i
        sample_end = min(SAMPLE_CHUNK_SIZE_STU * (i+1), n_stu)
        W = read_bed(stu_bed_filepref_chunk_list[i], bed_store='memory', dtype=np.int8)[0]
        elapse_load += time.time() - t0

        if method == 'oadp':
            assert (U is not None) & (s is not None) and (V is not None)
        elif method == 'adp':
            assert (XTX is not None) and (X is not None)
        elif method == 'ap':
            assert U is not None
        elif method == 'sp':
            assert U is not None
        else:
            logging.error(Method + ' is not one of sp, ap, adp, or oadp.')
            assert False

        for i in range(W.shape[1]):
            t0 = time.time()
            logging.debug('Extracting one row...')
            w = W[:,i].astype(np.float64).reshape((-1,1))
            logging.debug('Standardizing...')
            standardize(w, X_mean, X_std, miss=3)
            elapse_standardize += time.time() - t0
            t0 = time.time()
            logging.debug('Method...')
            if method == 'oadp':
                pcs_stu[sample_start + i, :] = oadp(U, s, V, w, dim_ref, dim_stu, dim_stu_high)
            elif method == 'adp':
                pcs_stu[sample_start + i, :] = adp(XTX, X, w, pcs_ref, dim_stu=dim_stu)
            elif method == 'ap':
                pcs_stu[sample_start + i, :] = w.T @ (U[:,:dim_ref])
            elif method == 'sp':
                pcs_stu[sample_start + i, :] = w.T @ (U[:,:dim_ref])
            else:
                logging.error(Method + ' is not one of sp, ap, adp, or oadp.')
                assert False
            elapse_method += time.time() - t0
        logging.info('Finished analyzing ' + str(sample_end) + ' samples.')

    logging.info('Finished analyzing all study samples.')
    print('='*80)
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

def pca(X_filepref, W_filepref, out_pref, method, path_tmp,
        dim_ref=DIM_REF, dim_stu=DIM_STU, dim_stu_high=DIM_STU_HIGH, hdpca_n_spikes_max=HDPCA_N_SPIKES_MAX,
        use_memmap=False, load_saved_ref_decomp=True):

    Xmnsd_filename = out_pref + '_mnsd.dat'
    s_filename = out_pref + '_s.dat'
    V_filename = out_pref + '_V.dat'
    U_filename = out_pref + '_U.dat'
    pcsref_filename = out_pref + '_ref.pcs'
    ref_decomp_filenames = [Xmnsd_filename, s_filename, V_filename, U_filename, pcsref_filename]
    ref_decomp_allexist = all([os.path.isfile(filename) for filename in ref_decomp_filenames])

    if ref_decomp_allexist and load_saved_ref_decomp:
        logging.info('Loading mean, sd, and SVD of ref data...')
        Xmnsd = np.loadtxt(Xmnsd_filename)
        X_mean = Xmnsd[:,0].reshape((-1,1))
        X_std = Xmnsd[:,1].reshape((-1,1))
        s = np.loadtxt(s_filename)
        V = np.loadtxt(V_filename)
        pcs_ref = np.loadtxt(pcsref_filename)
        U = np.loadtxt(U_filename)
    else:
        logging.info('Reading reference samples...')
        if use_memmap:
            mem_out_type = 'memmap_C'
        else:
            mem_out_type = 'memory'
        X, X_bim, X_fam = read_bed(X_filepref, bed_store=mem_out_type, dtype=np.float32)
        W_bim, W_fam = read_bed(W_filepref, bed_store=None)[1:3]
        assert W_bim.equals(X_bim)

        logging.info('Standardizing reference data...')
        X_mean, X_std = standardize(X)
        np.savetxt(Xmnsd_filename, np.hstack((X_mean, X_std)), fmt=NP_OUTPUT_FMT)

        s, V, XTX = pca_ref(X)
        V = V[:, :dim_stu_high]
        pcs_ref = V[:, :dim_ref] * s[:dim_ref]
        np.savetxt(s_filename, s, fmt=NP_OUTPUT_FMT)
        np.savetxt(V_filename, V, fmt=NP_OUTPUT_FMT)
        np.savetxt(pcsref_filename, pcs_ref, fmt=NP_OUTPUT_FMT)
        logging.info('Reference PC scores saved to ' + out_pref + '_ref.pcs')

        logging.info('Calculating PC loadings...')
        U = X @ (V[:,:dim_stu_high] / s[:dim_stu_high])
        np.savetxt(U_filename, U, fmt=NP_OUTPUT_FMT)

    p_ref = X_mean.shape[0]
    n_ref = V.shape[0]

    if method == 'ap':
        adj_hdpc_shrinkage(U, s, p_ref, n_ref, dim_ref)

    pcs_stu = pca_stu(W_filepref, X_mean, X_std, method=method, path_tmp=path_tmp,
                      U=U, s=s, V=V, pcs_ref=pcs_ref,
                      dim_ref=dim_ref, dim_stu=dim_stu, dim_stu_high=dim_stu_high)
    pcs_stu_filename = out_pref + '_stu_' + method +'.pcs'
    np.savetxt(pcs_stu_filename, pcs_stu, fmt=NP_OUTPUT_FMT, delimiter='\t')
    logging.info('Study PC scores saved to ' + pcs_stu_filename)
    return pcs_ref, pcs_stu

def run_pca(pref_ref, pref_stu, popu_filename_ref=None, popu_ref_k=None, method='oadp', dim_ref=DIM_REF, dim_stu=DIM_STU, dim_stu_high=DIM_STU_HIGH, use_memmap=False, load_saved_ref_decomp=True, log_level='info'):
    t0 = time.time()
    dir_ref = os.path.dirname(pref_ref)
    dir_stu = os.path.dirname(pref_stu)
    path_tmp = os.path.join(dir_stu, DIR_TMP)
    assert 2 <= dim_ref <= dim_stu <= dim_stu_high
    log = create_logger(pref_stu, log_level)
    assert (popu_filename_ref is None) != (popu_ref_k is None)
    logging.info('Reference data: ' + pref_ref)
    logging.info('Study data: ' + pref_stu)
    logging.debug('Tmp path: ' + path_tmp)
    subprocess.run(['mkdir', '-p', path_tmp])
    # Intersect ref and stu snps
    pref_ref_commsnpsrefal, pref_stu_commsnpsrefal = intersect_ref_stu_snps(pref_ref, pref_stu, path_tmp)

    W_bim, W_fam = read_bed(pref_stu_commsnpsrefal, bed_store=None)[1:3]

    # PCA on study individuals
    pcs_ref, pcs_stu = pca(pref_ref_commsnpsrefal, pref_stu_commsnpsrefal, pref_stu, method, path_tmp, dim_ref, dim_stu, dim_stu_high, use_memmap=use_memmap, load_saved_ref_decomp=load_saved_ref_decomp)

    # Read or predict ref populations
    if popu_filename_ref is not None:
        logging.info('Reading reference population from ' + popu_filename_ref)
        popu_ref = pd.read_table(popu_filename_ref, header=None).iloc[:,2]
    else:
        logging.info('Predicting reference population...')
        popu_ref = KMeans(n_clusters=popu_ref_k).fit_predict(pcs_ref)
        popu_ref_df = pd.DataFrame({'fid':X_fam['fid'], 'iid':X_fam['iid'], 'popu':popu_ref})
        popu_ref_df.to_csv(pref_ref+'_pred.popu', sep=DELIMITER, header=False, index=False)
        logging.info('Reference population prediction saved to ' + pref_ref+'_pred.popu')

    # Predict stu population
    popu_stu_pred = pred_popu_stu(pcs_ref, popu_ref, pcs_stu)
    popu_stu_pred_df = pd.DataFrame({'fid':W_fam['fid'], 'iid':W_fam['iid'], 'popu':popu_stu_pred})
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

    print('Total runtime: ' + str(time.time() - t0))
    return pcs_ref, pcs_stu, popu_ref, popu_stu_pred
