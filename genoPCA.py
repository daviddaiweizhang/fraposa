import numpy as np
import subprocess as sp
import os
from os import path as op
import math


import procdata as pd

# User input
KK = 10
L = 100
M = 0.01
GGS_DIR = '../data/ggs'
DATA_DIR = '../data/ggs_out'

# Program setting
INT_TYPE = np.int8
INT_FMT = '%d'
FLOAT_TYPE = np.float32
AXIS = 1  # rows are SNPs, cols are indvs
DELIM = '\t'

# For my convenience
TRACE_DIR = '../data/laser'
TRACE_VER = 'ori'

# Process the info above
GGS_DIR = op.realpath(GGS_DIR)
GGS_EXE = GGS_DIR + '/' + 'ggs'
TRACE_DIR = op.realpath(TRACE_DIR)
TRACE_EXE = TRACE_DIR + '/' + 'trace' + '.' + TRACE_VER


def ggs(outFile, KK, c, G, L, M, exe=GGS_EXE):
    outFile = op.realpath(outFile)
    exe = op.realpath(exe)
    cmd = [
        exe,
        '-o', outFile,
        '-K', str(KK),
        '-c', str(c),
        '-G', str(G),
        '-L', str(L),
        '-M', str(M)
    ]
    # Use devnull to suppress output
    devnull = open(os.devnull, 'w')
    sp.call(cmd, stdout=devnull)


def ggs_smart(prefix, p, n, KK=KK, L=L, M=M, data_dir=DATA_DIR):
    data_dir = op.realpath(data_dir)
    outFile = data_dir + '/' + prefix + '_' + str(p) + '_' + str(n) + '.ggs'
    assert(p % L == 0)
    G = p // L
    assert(n % (KK * KK) == 0)
    c = n // (KK * KK) * 2
    ggs(outFile, KK, c, G, L, M)


def ggs2dup(ggsFile, dupFile):
    ''' Convert the haploid genotype simulated by ggs to a duploid genotype. '''
    # print('Creating duploid genotype...')
    nCols = pd.getNCols(ggsFile) - 1  # The first column is the row names
    assert(nCols % 2 == 0)
    with open(ggsFile, 'r') as ggsF, open(dupFile, 'w') as dupF:
        next(ggsF)  # skip the header
        for i, line in enumerate(ggsF):
            lst = line.split()
            lst = list(map(int, lst[1:]))  # skip the row name
            lst = [x + y for x, y in zip(lst[0::2], lst[1::2])]
            lst = list(map(str, lst))
            st = '\t'.join(lst) + '\n'
            dupF.write(st)
            # if i % 100000 == 0:
            #     print('Finished line ' + str(i))
    # print('Finished!')


def ggs2dup_smart(prefix, p, n, s=0, data_dir=DATA_DIR):
    data_dir = op.realpath(data_dir)
    prefix = data_dir + '/' + prefix + '_' + str(p) + '_' + str(n)
    ggsFile = prefix + '.ggs'
    dupFile = prefix + '_' + str(s) + '.geno'
    ggs2dup(ggsFile, dupFile)


def getEvenCols_smart(prefix, p, nplusm, s=0, data_dir=DATA_DIR):
    assert(nplusm % 2 == 0)
    n = nplusm // 2
    prefix = data_dir + '/' + prefix + '_' + str(p) + '_'
    prefix = op.realpath(prefix)
    inFile = prefix + str(nplusm) + '_' + str(s) + '.geno'
    outFile = prefix + str(n) + '_0.geno'
    othFile = prefix + str(n) + '_1.geno'
    pd.getEvenCols(inFile, outFile, othFile)
    return outFile, othFile


def genGenoPair(prefix, p, n, KK=KK, L=L, M=M, data_dir=DATA_DIR, weight=None):
    nplusm = n * 2
    ggs_smart(prefix, p, nplusm)
    ggs2dup_smart(prefix, p, nplusm)
    ref_geno, stu_geno = getEvenCols_smart(prefix, p, nplusm)
    if weight is not None:
        pd.getWeightedCols(ref_geno, weight)
        pd.getWeightedCols(stu_geno, weight)
    return ref_geno, stu_geno


def analyze_stu_classic(X, XTX, Y, k, stu_coord=None, floatType=FLOAT_TYPE):
    p, n = X.shape
    q, m = Y.shape
    PCScores = np.zeros((m, k))
    for i in range(m):
        y = Y[:, i]
        XTy = np.dot(X.T, y)
        XTXAug = np.zeros((n + 1, n + 1), dtype=floatType)
        XTXAug[:-1, :-1] = XTX
        XTXAug[:-1, -1] = XTy
        XTXAug[-1, :-1] = XTy
        XTXAug[-1, -1] = np.dot(y, y)
        dAug, VAug = np.linalg.eigh(XTXAug)
        dAug = np.sqrt(abs(dAug))
        PCSdI = dAug[-k:]
        PCSdI = np.flipud(PCSdI)
        PCScoreI = VAug[-1, -k:]
        PCScoreI = np.flipud(PCScoreI)
        PCScoreI *= PCSdI
        PCScores[i, :] = PCScoreI
    # Save the PC scores to file
    if stu_coord is not None:
        np.savetxt(stu_coord, PCScores, fmt='%10.5f', delimiter='\t')
    return PCScores


def XTXdV(X, k, ref_coord=None, ref_sd=None):
    # Get PC scores for the reference group
    XTX = np.dot(X.T, X)
    d, V = np.linalg.eigh(XTX)
    d = np.sqrt(abs(d))
    V_save = V[:, -k:]
    V_save = np.fliplr(V_save)
    d_save = d[-k:]
    d_save = np.flipud(d_save)
    V_save *= d_save  # Use this version of PCA
    if ref_coord is not None:
        np.savetxt(ref_sd, d_save, fmt='%10.5f', delimiter='\t')
    if ref_sd is not None:
        np.savetxt(ref_coord, V_save, fmt='%10.5f', delimiter='\t')
    return XTX, d_save, V_save


def classic_pca_smart(prefix, p, n, k, data_dir=DATA_DIR, intType=INT_TYPE, floatType=FLOAT_TYPE):
    prefix = data_dir + '/' + prefix + '_' + str(p) + '_' + str(n)
    ref_geno = prefix + '_0.geno'
    stu_geno = prefix + '_1.geno'
    meth = 'classic'
    ref_coord = prefix + '_' + str('ref') + '.coord'
    ref_sd = prefix + '_' + str('ref') + '.sd'
    stu_coord = prefix + '_' + str(meth) + '.coord'
    return classic_pca(ref_geno, stu_geno, ref_coord, ref_sd, stu_coord, k)


def genname(filetype, prefix, p, n, s=None, meth=None, data_dir=DATA_DIR, trace_dir=TRACE_DIR, ggs_dir=GGS_DIR):
    data_dir = op.realpath(data_dir)
    trace_dir = op.realpath(trace_dir)
    ggs_dir = op.realpath(ggs_dir)
    if filetype == 'coord':
        filename = data_dir + '/' + prefix + '_' + \
            str(p) + '_' + str(n) + '_' + meth + '.coord'
    elif filetype == 'geno':
        filename = data_dir + '/' + prefix + '_' + \
            str(p) + '_' + str(n) + '_' + str(s) + '.geno'
    elif filetype == 'trace_geno':
        filename = data_dir + '/' + prefix + '_' + \
            str(p) + '_' + str(n) + '_' + str(s) + '.trce'
    elif filetype == 'trace_out_prefix':
        filename = data_dir + '/' + prefix + '_' + str(p) + '_' + str(n)
    elif filetype == 'site_src':
        filename = ggs_dir + '/' + str(p) + '.site'
    elif filetype == 'site_dest':
        filename = data_dir + '/' + prefix + '_' + \
            str(p) + '_' + str(n) + '_' + str(s) + '.site'
    elif filetype == 'trace_ref_out':
        filename = genname('trace_out_prefix', prefix, p, n)
        filename = filename + '.RefPC.coord'
    elif filetype == 'trace_stu_out':
        filename = genname('trace_out_prefix', prefix, p, n)
        filename = filename + '.ProPC.coord'
    else:
        raise Exception("File type not recognized.")
    return filename


def genTraceConf(prefix, p, n, dim=4, dim_high=20, trace_dir=TRACE_DIR):
    conName = 'trace.conf'
    confFile = trace_dir + '/' + conName
    with open(confFile, 'w') as conf:
        stu_file = genname('trace_geno', prefix, p, n, s=1)
        ref_file = genname('trace_geno', prefix, p, n, s=0)
        out_prefix = genname('trace_out_prefix', prefix, p, n)
        conf.write('STUDY_FILE' + '\t' + stu_file + '\n')
        conf.write('GENO_FILE' + '\t' + ref_file + '\n')
        conf.write('OUT_PREFIX' + '\t' + out_prefix + '\n')
        conf.write('DIM' + '\t' + str(dim) + '\n')
        conf.write('DIM_HIGH' + '\t' + str(dim_high) + '\n')
    return confFile


def geno2trace(genoFile, traceFile):
    X = np.loadtxt(genoFile, dtype=INT_TYPE)
    p, n = X.shape
    header = np.matrix([99] * n * 2).reshape((2, n))
    X = np.concatenate((header, X))
    X = X.T
    np.savetxt(traceFile, X, delimiter='\t', fmt='%d')


def center(X, axis, X_mean=None):
    if X_mean is None:
        X_mean = np.mean(X, axis=axis, keepdims=True)
    X -= X_mean
    return X_mean


def normalize(X, axis, X_norm=None):
    if X_norm is None:
        p, n = np.shape(X)
        # The Frobenius norm does not take into account of the length of the vector
        X_norm = np.linalg.norm(X, axis=axis, keepdims=True) / np.sqrt(n)
    X_norm[X_norm == 0] = 1
    X /= X_norm
    # for i, s in enumerate(X_norm):
    #     if s != 0:
    #         X[i, :] /= s
    return X_norm


def trace(prefix, p, n, trace_exe=TRACE_EXE, ggs_dir=GGS_DIR):
    traceConfFile = genTraceConf(prefix, p, n)
    trace_site_src = genname('site_src', prefix, p, n)
    trace_site_dest_ref = genname('site_dest', prefix, p, n, s=0)
    trace_site_dest_stu = genname('site_dest', prefix, p, n, s=1)
    sp.call(['cp', trace_site_src, trace_site_dest_ref])
    sp.call(['cp', trace_site_src, trace_site_dest_stu])
    sp.call([trace_exe, '-p', traceConfFile])
    trace_ref_output = genname('trace_ref_out', prefix, p, n)
    trace_stu_output = genname('trace_stu_out', prefix, p, n)
    return trace_ref_output, trace_stu_output


def load_trace_output(trace_output, refstu, k):
    if refstu == 'ref':
        skipcols = 2
    elif refstu == 'stu':
        skipcols = 5
    else:
        raise Exception('Please enter ref or stu only.')
    allcols = pd.getNCols(trace_output)
    skiprows = 1
    usecols = list(range(skipcols, allcols))
    trace_coord = np.loadtxt(
        trace_output, skiprows=skiprows, usecols=usecols)
    return trace_coord


def load_trace_output_smart(prefix, p, n, refstu, k):
    if refstu == 'ref':
        trace_output = genname('trace_ref_out', prefix, p, n)
    elif refstu == 'stu':
        trace_output = genname('trace_stu_out', prefix, p, n)
    else:
        raise Exception('Please enter ref or stu only.')
    return load_trace_output(trace_output, refstu, k)


def procrustes(X, Y):
    # input: X, Y
    # output: rho, A, b, (t)
    # Create a copy, rather a new reference to the old matrix
    Xc = np.copy(X)
    Yc = np.copy(Y)
    Xm = center(Xc, axis=0)
    Ym = center(Yc, axis=0)
    C = np.dot(Yc.T, Xc)
    U, s, V = np.linalg.svd(C, full_matrices=False)
    trXX = np.matrix.trace(np.dot(Xc.T, Xc))
    trYY = np.matrix.trace(np.dot(Yc.T, Yc))
    trS = sum(s)
    A = np.dot(V, U.T)
    rho = trS / trXX
    b = Ym - rho * np.dot(Xm, A)
    # # Compute similarity t
    # Xnew = rho * np.dot(X, A) + b
    # Z = Y - Xnew
    # d = np.matrix.trace(np.dot(Z.T, Z))
    # D = d / trYY
    # t = np.sqrt(1 - D)
    return rho, A, b


def standardize(X_dup, Y_dup, axis, floatType=FLOAT_TYPE):
    # Preprocess datasets
    X = floatType(X_dup)
    X_mean = center(X, axis=axis)
    X_norm = normalize(X, axis=axis)
    Y = floatType(Y_dup)
    center(Y, axis=axis, X_mean=X_mean)
    normalize(Y, axis=axis, X_norm=X_norm)
    return X, Y


def loadXYdup(ref_geno, stu_geno, k, intType=INT_TYPE):
    # Load datasets
    # Input format:
    # rows are SNPs, cols are indvs (axis = 1)
    X_dup = np.loadtxt(ref_geno, dtype=intType)
    Y_dup = np.loadtxt(stu_geno, dtype=intType)
    p, n = X_dup.shape
    q, m = Y_dup.shape
    assert(p == q)
    assert(k <= n + 1)
    return X_dup, Y_dup


def classic_pca(ref_geno, stu_geno, ref_coord, ref_sd, stu_coord, k, intType=INT_TYPE, floatType=FLOAT_TYPE):
    X_dup, Y_dup = loadXYdup(ref_geno, stu_geno, k)
    X, Y = standardize(X_dup, Y_dup, axis=AXIS, floatType=FLOAT_TYPE)
    XTX, d, V = XTXdV(X, k, ref_coord, ref_sd)
    YVd_classic = analyze_stu_classic(X, XTX, Y, k)
    # This line has problems
    YVd_proj = Y.T.dot(X).dot(V).dot(np.diag(1 / d))
    return V, YVd_classic, YVd_proj
