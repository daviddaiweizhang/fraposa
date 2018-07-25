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
import multiprocessing as mp

PYTHONHASHSEED = 24
DIM_REF = 4
DIM_STU = 20
DIM_STU_HIGH = DIM_STU * 2
N_NEIGHBORS=23
HDPCA_N_SPIKES_MAX = 18
HDPCA_N_SPIKES = DIM_REF
SAMPLE_CHUNK_SIZE_STU = 5000
SAMPLE_SPLIT_PREF_LEN = 4
PROCRUSTES_NITER_MAX = 10000
PROCRUSTES_EPSILON_MIN = 1e-6
NUM_CORES = mp.cpu_count()


PLOT_ALPHA_REF=0.4
PLOT_ALPHA_STU=0.2
PLOT_MARKERS = ['.', '+', 'x', 'd', '*', 's']
LOG_LEVEL = 'info'

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
    log_dir = os.path.dirname(prefix) + '/logs/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    filename = log_dir + os.path.basename(prefix) + '.' + str(round(time.time())) + '.log'
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

def geocenter_coordinate(X, X_ctr):
    X_ctr = np.array(X_ctr)
    p = X.shape[1]
    X_ctr_unique = np.unique(X_ctr)
    X_ctr_unique_n = len(X_ctr_unique)
    X_ctr_unique_coord = np.zeros((X_ctr_unique_n, p))
    for i,ctr in enumerate(X_ctr_unique):
        X_ctr_unique_coord[i] = np.mean(X[X_ctr == ctr], axis=0)
    X_ctr_coord_dic = {X_ctr_unique[i] : X_ctr_unique_coord[i] for i in range(X_ctr_unique_n)}
    return X_ctr_coord_dic

def geocenter_similarity(Y, Y_ctr, X, X_ctr):
    assert X.shape[1] == Y.shape[1]
    X_ctr_coord_dic = geocenter_coordinate(X, X_ctr)
    Y_ctr_coord_dic = geocenter_coordinate(Y, Y_ctr)
    dist = 0
    for ctr in Y_ctr_coord_dic:
        dist += np.sum((Y_ctr_coord_dic[ctr] - X_ctr_coord_dic[ctr])**2)
    return dist

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

def intersect_ref_stu_snps(pref_ref, pref_stu):
    snps_are_identical = filecmp.cmp(pref_ref+'.bim', pref_stu+'.bim')
    if snps_are_identical:
        logging.info('SNPs and alleles in reference and study samples are identical')
        return pref_ref, pref_stu
    else:
        logging.error('Error: SNPs and alleles in reference and study samples are not identical')
        assert False
        logging.info('Intersecting SNPs in reference and study samples...')
        bashout = subprocess.run(
            ['bash', 'intersect_bed.sh', pref_ref, pref_stu],
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

def bed2trace(filepref, missing=3):
    log = create_logger(filepref, 'info')
    bed, bim, fam = read_bed(filepref)
    for idx, x in np.ndenumerate(bed):
        if x == missing:
            bed[idx] = -9
    bed_df = pd.DataFrame(bed.T)
    tracegeno = pd.concat([fam[['fid', 'iid']], bed_df], axis=1, join_axes=[fam.index])
    tracegeno.to_csv(filepref+'.geno', sep='\t', header=False, index=False)

    tracesite = pd.DataFrame({
        'CHROM': bim['chrom'],
        'POS': bim['pos'],
        'ID': bim.index.values,
        'REF': bim['a1'],
        'ALT': bim['a2']
    })
    tracesite = tracesite[['CHROM', 'POS', 'ID', 'REF', 'ALT']]
    tracesite.to_csv(filepref+'.site', sep='\t', header=True, index=False)

def trace2bed(trace_filepref, missing=3):
    bed_filepref = trace_filepref

    with open(trace_filepref+'.geno', 'r') as f:
        ncols = len(f.readline().split())
    bed = np.loadtxt(trace_filepref+'.geno', dtype=np.int8, usecols=range(2, ncols)).T
    bed[bed == -9] = missing
    bed = 2 - bed
    with PyPlink(bed_filepref, 'w') as pyp:
        for row in bed:
            pyp.write_genotypes(row)

    fam = pd.read_table(trace_filepref+'.geno', usecols=range(0, 2), header=None)
    popu = pd.concat([fam, fam.iloc[:,0]], axis=1)
    popu.to_csv(bed_filepref+'.popu', sep='\t', header=False, index=False)
    nrows = fam.shape[0]
    filler = pd.DataFrame(np.zeros((nrows, 4), dtype=np.int8))
    fam = pd.concat([fam, filler], axis=1)
    fam.to_csv(bed_filepref+'.fam', sep='\t', header=False, index=False)

    site = pd.read_table(trace_filepref+'.site')
    bim = site[['CHR', 'ID', 'POS', 'REF', 'ALT']]
    bim.insert(2, 'cm', 0)
    bim.to_csv(bed_filepref+'.bim', sep='\t', header=False, index=False)

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

def eig_ref(X):
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


def get_pcs_stu_this(output, i, W, X_mean, X_std, U, s, V, method, dim_ref=DIM_REF, dim_stu=DIM_STU, dim_stu_high=DIM_STU_HIGH):
    w = W[:,i].astype(np.float64).reshape((-1,1))
    standardize(w, X_mean, X_std, miss=3)
    if method == 'oadp':
        pcs_stu_this = oadp(U, s, V, w, dim_ref, dim_stu, dim_stu_high)
    # elif method == 'adp':
    #     pcs_stu_this = adp(XTX, X, w, pcs_ref, dim_stu=dim_stu)
    elif method == 'ap':
        pcs_stu_this = w.T @ (U[:,:dim_ref])
    elif method == 'sp':
        pcs_stu_this = w.T @ (U[:,:dim_ref])
    else:
        assert False
    output.put((i, pcs_stu_this))


def pred_popu_stu(pcs_ref, popu_ref, pcs_stu, out_filename=None, rowhead_df=None):
    logging.info('Predicting populations for study individuals...')
    popu_list = np.sort(np.unique(popu_ref))
    popu_dic = {popu_list[i] : i for i in range(len(popu_list))}
    knn = KNeighborsClassifier(n_neighbors=N_NEIGHBORS)
    knn.fit(pcs_ref, popu_ref)
    popu_stu_pred = knn.predict(pcs_stu)
    popu_stu_proba_list = knn.predict_proba(pcs_stu)
    popu_stu_proba = [popu_stu_proba_list[i, popu_dic[popu_stu_pred[i]]] for i in range(pcs_stu.shape[0])]
    popu_stu_dist = knn.kneighbors(pcs_stu)[0][:,-1]
    popu_stu_dist = np.round(popu_stu_dist, 3)
    if out_filename is not None:
        popuproba_df = pd.DataFrame({'popu':popu_stu_pred, 'proba':popu_stu_proba, 'dist':popu_stu_dist})
        popuproba_df = popuproba_df[['popu', 'proba', 'dist']]
        popu_stu_pred_df = pd.concat([rowhead_df, popuproba_df], axis=1)
        popu_stu_pred_df.to_csv(out_filename, sep=DELIMITER, header=False, index=False)
        logging.info('Predicted study populations saved to ' + out_filename)
    return popu_stu_pred, popu_stu_proba, popu_stu_dist

def pred_popu_ref(pcs_ref, n_clusters, out_filename=None, rowhead_df=None):
    popu_ref = KMeans(n_clusters=n_clusters).fit_predict(pcs_ref)
    if out_filename is not None:
        popu_ref_df = pd.DataFrame({'popu':popu_ref})
        popu_ref_df = pd.concat([rowhead_df, popu_ref_df], axis=1)
        popu_ref_df.to_csv(out_filename, sep=DELIMITER, header=False, index=False)
        print('Reference clustering saved to ' + out_filename)
    popu_ref = np.array([str(pp) for pp in popu_ref])
    return popu_ref

def load_pcs(pref, methods):
    logging.info('Loading existing reference and study PC scores...')
    pcs_ref = np.loadtxt(pref+'_ref.pcs')
    pcs_stu_list = []
    for mth in methods:
        pcs_stu_filename = pref + '_stu_' + mth + '.pcs'
        pcs_stu_this = np.loadtxt(pcs_stu_filename)
        pcs_stu_list += [pcs_stu_this]
    return pcs_ref, pcs_stu_list

def hdpca_shrinkage(U, s, p_ref, n_ref, hdpca_n_spikes=None, hdpca_n_spikes_max=HDPCA_N_SPIKES_MAX):
    if hdpca_n_spikes is None:
        hdpc_est_result  = hdpc_est(s**2, p_ref, n_ref, n_spikes_max=hdpca_n_spikes_max)
    else:
        hdpc_est_result  = hdpc_est(s**2, p_ref, n_ref, n_spikes=hdpca_n_spikes)
    shrinkage = np.array(hdpc_est_result[-1])
    return shrinkage


def plot_pcs(pcs_ref, pcs_stu=None, popu_ref=None, popu_stu=None, method=None, out_pref=None, markers=PLOT_MARKERS, alpha_ref=PLOT_ALPHA_REF, alpha_stu=PLOT_ALPHA_STU, plot_lim=None, plot_dim=float('inf'), plot_size=None, plot_title=None, plot_color_stu=None, plot_legend=True, plot_centers=False):
    n_ref, dim_ref = pcs_ref.shape
    if pcs_stu is None:
        n_stu = 0
    else:
        n_stu = pcs_stu.shape[0]
    if method is None:
        method = 'stu'
    if popu_ref is None:
        popu_ref = np.array(['ref'] * n_ref)
    else:
        popu_ref = np.array(popu_ref, dtype='str')
    if popu_stu is None:
        popu_stu = np.array([popu_ref[0]] * n_stu)
    else:
        popu_stu = np.array(popu_stu, dtype='str')

    # Get the unique populations and assign plotting colors to them
    popu_unique = sorted(list(set(np.concatenate((popu_ref, popu_stu)))))
    popu_unique = sorted(popu_unique, key=len)
    popu_n = len(popu_unique)
    colormap = plt.get_cmap('tab10')
    popu2color = dict(zip(popu_unique, colormap(range(popu_n))))

    # Plotting may need to change the signs of PC scores
    # Make a copy to avoid modifying original arrays
    pcs_ref = np.copy(pcs_ref)
    if pcs_stu is not None:
        pcs_stu = np.copy(pcs_stu)
    if plot_lim is not None:
        plot_lim = np.copy(plot_lim)

    # Make sure the same population has the same PC score "shape" when analyzed by different methods
    geocenter_coord = geocenter_coordinate(pcs_ref, popu_ref).values()
    geocenter_coord = list(geocenter_coord)
    geocenter_dist = np.array([np.linalg.norm(v) for v in geocenter_coord])
    geocenter_farthest_sign = np.sign(geocenter_coord[geocenter_dist.argmax()])
    for i in range(dim_ref):
        sign_this = geocenter_farthest_sign[i]
        if sign_this < 0:
            pcs_ref[:,i] *= sign_this
            if pcs_stu is not None:
                pcs_stu[:,i] *= sign_this
            if plot_lim is not None:
                plot_lim[:,i] *= sign_this
                plot_lim[:,i] = plot_lim[:,i][::-1]

    n_subplot = int(min(dim_ref, plot_dim) / 2)
    fig, ax = plt.subplots(ncols=n_subplot)
    for j in range(n_subplot):
        plt.subplot(1, n_subplot, j+1)
        plt.xlabel('PC' + str(j*2+1))
        plt.ylabel('PC' + str(j*2+2))
        for i,popu in enumerate(popu_unique):
            ref_is_this_popu = popu_ref == popu
            pcs_ref_this_popu = pcs_ref[ref_is_this_popu, (j*2):(j*2+2)]
            plot_color_this = popu2color[popu]
            if plot_centers:
                label = None
            else:
                label = str(popu)
            plt.scatter(pcs_ref_this_popu[:,0], pcs_ref_this_popu[:,1], marker=markers[-1], alpha=alpha_ref, color=plot_color_this, label=label)
            if plot_centers:
                pcs_ref_this_popu_mean = np.mean(pcs_ref_this_popu, axis=0)
                plt.scatter(pcs_ref_this_popu_mean[0], pcs_ref_this_popu_mean[1], marker=markers[-2], color=plot_color_this, edgecolor='xkcd:grey', s=300, label=str(popu))
        if pcs_stu is not None:
            if plot_color_stu is None:
                plot_color_stu_list = np.array([popu2color[popu_this] for popu_this in popu_stu], dtype=np.object)
            else:
                plot_color_stu_list = np.array([plot_color_stu] * pcs_stu.shape[0], dtype=np.object)
            a = 5
            for i in range(a):
                if i == 0:
                    label = 'stu (' + method + ')'
                else:
                    label = None
                indiv_shuffled_this = np.arange(pcs_stu.shape[0]) % a == i
                plt.scatter(pcs_stu[indiv_shuffled_this, j*2],
                            pcs_stu[indiv_shuffled_this, j*2+1],
                            color=plot_color_stu_list[indiv_shuffled_this].tolist(),
                            marker=markers[0], alpha=alpha_stu, label=label)
            if plot_centers:
                for i,popu in enumerate(popu_unique):
                    stu_is_this_popu = popu_stu == popu
                    pcs_stu_this_popu = pcs_stu[stu_is_this_popu, (j*2):(j*2+2)]
                    pcs_stu_this_popu_mean = np.mean(pcs_stu_this_popu, axis=0)
                    plt.scatter(pcs_stu_this_popu_mean[0], pcs_stu_this_popu_mean[1], marker=markers[-2], color='xkcd:grey', s=100)
        if plot_lim is not None:
            plt.xlim(plot_lim[:,j*2])
            plt.ylim(plot_lim[:,j*2+1])
    if plot_legend:
        plt.legend()
    if plot_title is not None:
        plt.title(str(method)+' '+plot_title, fontsize=30)
    plt.tight_layout()
    if plot_size is not None:
        fig.set_size_inches(plot_size)
    if out_pref is not None:
        fig_filename = out_pref + '_' + method + '.png'
        plt.savefig(fig_filename, dpi=300)
        logging.info('PC plots saved to ' + fig_filename)
    else:
        logging.info('No output path specified.')
        plt.show()
    plt.close('all')


def pca_stu(W, X_mean, X_std, method, 
            U=None, s=None, V=None, XTX=None, X=None, pcs_ref=None,
            dim_ref=DIM_REF, dim_stu=DIM_STU, dim_stu_high=DIM_STU_HIGH):
    p_ref = len(X_mean)
    p_stu, n_stu = W.shape
    pcs_stu = np.zeros((n_stu, dim_ref))

    elapse_standardize = 0.0
    elapse_method = 0.0

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

    for i in range(n_stu):
        t0 = time.time()
        w = W[:,i].astype(np.float64).reshape((-1,1))
        standardize(w, X_mean, X_std, miss=3)
        elapse_standardize += time.time() - t0
        t0 = time.time()
        logging.debug('Method...')
        if method == 'oadp':
            pcs_stu[i,:] = oadp(U, s, V, w, dim_ref, dim_stu, dim_stu_high)
        elif method =='adp':
            pcs_stu[i,:] = adp(XTX, X, w, pcs_ref, dim_stu=dim_stu)
        elif method =='ap':
            pcs_stu[i,:] = w.T @ (U[:,:dim_ref])
        elif method =='sp':
            pcs_stu[i,:] = w.T @ (U[:,:dim_ref])
        else:
            logging.error(Method + ' is not one of sp, ap, adp, or oadp.')
            assert False
        if (i+1) % 1000 == (0 or n_stu):
            logging.info('Finished ' + str(i+1) + ' study samples.')
        elapse_method += time.time() - t0

    logging.info('Runtimes: ')
    logging.info('Standardizing: ' + str(elapse_standardize))
    logging.info(method + ': ' + str(elapse_method))

    del W
    return pcs_stu

def pca_ref(X, dim_ref=DIM_REF, dim_stu_high=DIM_STU_HIGH):
    X_mean, X_std = standardize(X)
    s, V = eig_ref(X)[:2]
    V = V[:, :dim_stu_high]
    pcs_ref = V[:, :dim_ref] * s[:dim_ref]
    U = X @ (V[:,:dim_stu_high] / s[:dim_stu_high])
    return X_mean, X_std, U, s, V, pcs_ref

def pca(ref_filepref, stu_filepref, method,
        dim_ref=DIM_REF, dim_stu=DIM_STU, dim_stu_high=DIM_STU_HIGH, hdpca_n_spikes=None, hdpca_n_spikes_max=HDPCA_N_SPIKES_MAX,
        use_memmap=False, load_saved_ref_decomp=True):

    if stu_filepref is None:
        out_filepref = ref_filepref
    else:
        out_filepref = stu_filepref + '_sturef_' + os.path.basename(ref_filepref)
    Xmnsd_filename = ref_filepref + '_mnsd.dat'
    s_filename = ref_filepref + '_s.dat'
    V_filename = ref_filepref + '_V.dat'
    U_filename = ref_filepref + '_U.dat'
    pcs_ref_filename = ref_filepref + '_ref.pcs'
    ref_decomp_filenames = [Xmnsd_filename, s_filename, V_filename, U_filename, pcs_ref_filename]
    shrinkage_filename = ref_filepref + '_shrinkage.dat'
    ref_decomp_allexist = all([os.path.isfile(filename) for filename in ref_decomp_filenames])

    if ref_decomp_allexist and load_saved_ref_decomp:
        logging.info('Loading mean, sd, and SVD of ref data...')
        Xmnsd = np.loadtxt(Xmnsd_filename)
        X_mean = Xmnsd[:,0].reshape((-1,1))
        X_std = Xmnsd[:,1].reshape((-1,1))
        s = np.loadtxt(s_filename)
        V = np.loadtxt(V_filename)[:, :dim_stu_high]
        pcs_ref = np.loadtxt(pcs_ref_filename)[:, :dim_ref]
        U = np.loadtxt(U_filename)[:, :dim_stu_high]
    else:
        logging.info('Reading reference samples...')
        if use_memmap:
            mem_out_type = 'memmap_C'
        else:
            mem_out_type = 'memory'
        X, X_bim, X_fam = read_bed(ref_filepref, bed_store=mem_out_type, dtype=np.float32)

        logging.info('PCA on reference data...')
        X_mean, X_std, U, s, V, pcs_ref = pca_ref(X, dim_ref=dim_ref, dim_stu_high=dim_stu_high)
        np.savetxt(Xmnsd_filename, np.hstack((X_mean, X_std)), fmt=NP_OUTPUT_FMT)
        np.savetxt(s_filename, s, fmt=NP_OUTPUT_FMT)
        np.savetxt(V_filename, V, fmt=NP_OUTPUT_FMT)
        np.savetxt(pcs_ref_filename, pcs_ref, fmt=NP_OUTPUT_FMT)
        np.savetxt(U_filename, U, fmt=NP_OUTPUT_FMT)
        logging.info('Reference PC scores saved to ' + pcs_ref_filename)

    if method == 'ap':
        if os.path.isfile(shrinkage_filename) and load_saved_ref_decomp:
            shrinkage = np.loadtxt(shrinkage_filename)
        else:
            logging.info('Calculating PC shrinkage...')
            p_ref = X_mean.shape[0]
            n_ref = V.shape[0]
            shrinkage = hdpca_shrinkage(U, s, p_ref, n_ref, hdpca_n_spikes=hdpca_n_spikes)
            np.savetxt(shrinkage_filename, shrinkage, fmt=NP_OUTPUT_FMT)
        n_pc_adjusted = min(dim_ref, len(shrinkage))
        for i in range(n_pc_adjusted):
            U[:, i] /= shrinkage[i]

    if stu_filepref is None:
        return pcs_ref, None, pcs_ref_filename, None

    print('>'*30)
    logging.info('Predicting study PC scores (method: ' + method + ')...')
    W, W_bim, W_fam = read_bed(stu_filepref, bed_store='memory', dtype=np.int8)
    t0 = time.time()
    pcs_stu = pca_stu(W, X_mean, X_std, method=method,
                      U=U, s=s, V=V, pcs_ref=pcs_ref,
                      dim_ref=dim_ref, dim_stu=dim_stu, dim_stu_high=dim_stu_high)
    elapse_stu = time.time() - t0
    pcs_stu_filename = out_filepref + '_stu_' + method +'.pcs'
    np.savetxt(pcs_stu_filename, pcs_stu, fmt=NP_OUTPUT_FMT, delimiter='\t')
    logging.info('Study PC scores saved to ' + pcs_stu_filename)
    logging.info('Study time: ' + str(elapse_stu))
    print('<'*30)

    # pcs_stu_chunk_list = [None] * chunk_n_stu
    # pcs_stu_chunk_list = Parallel(n_jobs=NUM_CORES)(delayed(pca_stu_io)(i, stu_filepref_chunk_list, ref_filepref, path_tmp, method, X_mean, X_std, U, s, V, pcs_ref, dim_ref=DIM_REF, dim_stu=DIM_STU, dim_stu_high=DIM_STU_HIGH) for i in range(chunk_n_stu))

    # for i in range(chunk_n_stu):
    #     pcs_stu_chunk_list[i] = pca_stu_io(
    #             i, stu_filepref_chunk_list, ref_filepref, path_tmp, method,
    #             X_mean, X_std, U, s, V, pcs_ref,
    #             dim_ref=DIM_REF, dim_stu=DIM_STU, dim_stu_high=DIM_STU_HIGH)

    # pcs_stu = np.zeros((n_stu, dim_ref))
    # for i in range(chunk_n_stu):
    #     sample_start = i * SAMPLE_CHUNK_SIZE_STU
    #     sample_end = min([sample_start + SAMPLE_CHUNK_SIZE_STU, n_stu])
    #     pcs_stu[sample_start:sample_end,:] = pcs_stu_chunk_list[i]
    # out_filepref = stu_filepref + '_sturef_' + os.path.basename(ref_filepref)
    # logging.info('Study PC scores saved to ' + out_filepref)

    return pcs_ref, pcs_stu, pcs_ref_filename, pcs_stu_filename


def run_pca(pref_ref, pref_stu, method='oadp', dim_ref=DIM_REF, dim_stu=DIM_STU, dim_stu_high=DIM_STU_HIGH, use_memmap=False, load_saved_ref_decomp=True, log_level='info', plot_results=False, hdpca_n_spikes=None, n_clusters=None):
    print('='*30)
    t0 = time.time()
    base_ref = os.path.basename(pref_ref)
    if pref_stu is None:
        pref_out = pref_ref
    else:
        dir_stu = os.path.dirname(pref_stu)
        pref_out = pref_stu + '_sturef_' + base_ref
    log = create_logger(pref_out, log_level)

    assert 2 <= dim_ref <= dim_stu <= dim_stu_high
    logging.info('Using ' + str(NUM_CORES) + ' cores.')
    logging.info('Reference data: ' + pref_ref)
    if pref_stu is None:
        logging.info('PCA on reference samples only.')
        X_bim, X_fam = read_bed(pref_ref, bed_store=None)[1:3]
        pca_result = pca(pref_ref, None, method, dim_ref, dim_stu, dim_stu_high, use_memmap=use_memmap, load_saved_ref_decomp=load_saved_ref_decomp, hdpca_n_spikes=hdpca_n_spikes)
    else:
        logging.info('Study data: ' + pref_stu)
        logging.info('Method: ' + method)
        # Intersect ref and stu snps
        pref_ref_commsnpsrefal, pref_stu_commsnpsrefal = intersect_ref_stu_snps(pref_ref, pref_stu)
        W_bim, W_fam = read_bed(pref_stu_commsnpsrefal, bed_store=None)[1:3]
        X_bim, X_fam = read_bed(pref_ref_commsnpsrefal, bed_store=None)[1:3]
        # do PCA
        pca_result = pca(pref_ref_commsnpsrefal, pref_stu_commsnpsrefal, method, dim_ref, dim_stu, dim_stu_high, use_memmap=use_memmap, load_saved_ref_decomp=load_saved_ref_decomp, hdpca_n_spikes=hdpca_n_spikes)

    pcs_ref, pcs_stu, pcs_ref_filename, pcs_stu_filename = pca_result

    # Read or predict ref populations
    if n_clusters is None:
        popu_ref_filename = pref_ref + '.popu'
        logging.info('Reading reference population from ' + popu_ref_filename)
        popu_ref = pd.read_table(popu_ref_filename, header=None).iloc[:,2]
    else:
        popu_ref_filename = pref_ref + '_pred.popu'
        logging.info('Predicting reference population...')
        popu_ref = pred_popu_ref(pcs_ref, n_clusters, out_filename=popu_ref_filename, rowhead_df=X_fam[['fid','iid']])
        logging.info('Reference population prediction saved to ' + popu_ref_filename)

    # Predict stu population
    popu_stu_pred = None
    popu_stu_filename = None
    if  pref_stu is not None:
        popu_stu_filename = pref_out + '_pred_' + method + '.popu'
        popu_stu_pred, popu_stu_proba, popu_stu_dist = pred_popu_stu(pcs_ref, popu_ref, pcs_stu, out_filename=popu_stu_filename, rowhead_df=W_fam[['fid','iid']])
        logging.info('Study population prediction saved to ' + popu_stu_filename)

    # Plot PC scores
    if plot_results:
        plot_pcs(pcs_ref, pcs_stu, popu_ref, popu_stu_pred, method=method, out_pref=pref_out)

    logging.info('Total runtime: ' + str(time.time() - t0))
    return pcs_ref, pcs_stu, popu_ref, popu_stu_pred, pcs_ref_filename, pcs_stu_filename, popu_ref_filename, popu_stu_filename

def concat_files(inlist, out):
    with open(out, 'w') as outfile:
        for fname in inlist:
            with open(fname) as infile:
                outfile.write(infile.read())

def merge_array_results(ref_filepref, stu_filepref, method, n_chunks):
    ref_basepref = os.path.basename(ref_filepref)
    endcode_list = [str(i).zfill(SAMPLE_SPLIT_PREF_LEN) for i in range(n_chunks)]
    stu_filepref_list = [stu_filepref + '_' + endcode_list[i] + '_sturef_' + ref_basepref for i in range(n_chunks)]
    stu_pcs_filename_list = [fpref + '_stu_' + method + '.pcs' for fpref in stu_filepref_list]
    stu_popu_filename_list = [fpref + '_pred_' + method + '.popu' for fpref in stu_filepref_list]
    stu_fam_filename_list = [stu_filepref + '_' + endcode_list[i] + '.fam' for i in range(n_chunks)]
    stu_pcs_filename = stu_filepref + '_sturef_' + ref_basepref + '_stu_' + method + '.pcs'
    stu_popu_filename = stu_filepref + '_sturef_' + ref_basepref + '_pred_' + method + '.popu'
    stu_fam_filename = stu_filepref + '.fam'
    concat_files(stu_pcs_filename_list, stu_pcs_filename)
    concat_files(stu_popu_filename_list, stu_popu_filename)
    concat_files(stu_fam_filename_list, stu_fam_filename)
    # ref_pcs_filename = ref_filepref + '_ref.pcs'
    # ref_popu_filename = ref_filepref + '.popu'
    # ref_pcs = np.loadtxt(ref_pcs_filename)
    # stu_pcs = np.loadtxt(stu_pcs_filename)
    # ref_popu = np.loadtxt(ref_popu_filename, dtype=np.object)[:,2]
    # stu_popu = np.loadtxt(stu_popu_filename, dtype=np.object)[:,2]
    # plot_pcs(ref_pcs, stu_pcs, ref_popu, stu_popu, method, out_pref=stu_filepref)

def split_bed_indiv(filepref, n_chunks, i):
    assert i in range(n_chunks)
    basepref = os.path.basename(filepref)
    dirname = os.path.dirname(filepref)
    basepref_chunks = basepref + '_nchunks' + str(n_chunks)
    dirname_this = dirname + '/' + basepref_chunks
    filepref_this = dirname_this + '/' + basepref_chunks + '_' + str(i).zfill(SAMPLE_SPLIT_PREF_LEN)

    fam = np.loadtxt(filepref+'.fam', dtype='str')
    fam_this = np.array_split(fam, n_chunks)[i]
    fam_this_exists = False
    if os.path.exists(filepref_this+'.fam'):
        fam_this_existing = np.loadtxt(filepref_this+'.fam', dtype='str')
        if np.array_equal(fam_this, fam_this_existing):
            fam_this_exists = True
    if not fam_this_exists:
        os.makedirs(dirname_this, exist_ok=True)
        np.savetxt(filepref_this+'.fam', fam_this, delimiter=' ', fmt='%s')
        outerr = subprocess.run(
                ['plink', '--bfile', filepref, '--keep', filepref_this+'.fam', '--keep-allele-order', '--out', filepref_this, '--make-bed'],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        os.remove(filepref_this+'.log') # remove plink log
        assert filecmp.cmp(filepref_this+'.bim', filepref+'.bim')

    return filepref_this
