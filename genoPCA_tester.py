# Testing functions in genoPCA

# # Remove all user-defined variables
# import sys
# this = sys.modules[__name__]
# for n in dir():
#     if n[0] != '_':
#         delattr(this, n)

from os import path as op
import matplotlib.pyplot as plt
import subprocess as sp
import importlib
import numpy as np
import filecmp

import genoPCA as gp
import procdata as pd

importlib.reload(gp)

PREFIX = 'test'
P = 1000
N = 100

NPLUSM = N * 2
K = 4
DATA_DIR = '../data/ggs_out'
INT_TYPE = np.int8
INT_FMT = '%d'
FLOAT_TYPE = np.float32
DELIM = '\t'

TRACE_DIR = '../data/laser'
TRACE_DIR = op.realpath(TRACE_DIR)
TRACE_EXE = TRACE_DIR + '/' + 'trace.ori'

# sp.call(['rm', op.realpath(DATA_DIR) + '/test*'])


def test_ggs():
    print('Testing ggs...')
    K = 2
    c = 2
    G = 3
    L = 4
    M = 10
    data_dir = '../data/ggs_out'
    prefix = 'test'
    p = 1000
    n = 100
    outFile = data_dir + '/' + prefix
    outFile += '_' + str(p) + '_' + str(n)
    outFile += '.ggs'
    gp.ggs(outFile, K, c, G, L, M)
    print('Passed.')


def test_ggs_smart():
    print('Testing smart ggs...')
    prefix = 'test'
    p = 1000
    n = 100
    gp.ggs_smart(prefix, p, n)
    print('Passed.')


def test_ggs2dup():
    print('Testing ggs2dup...')
    data_dir = '../data/ggs_out'
    prefix = 'test'
    p = 1000
    n = 100
    prefix = data_dir + '/' + prefix + '_' + \
        str(p) + '_' + str(n)
    ggsFile = prefix + '.ggs'
    dupFile = prefix + '.geno'
    gp.ggs2dup(ggsFile, dupFile)
    print('Passed.')


def test_ggs2dup_smart():
    print('Testing ggs2dup_smart')
    prefix = 'test'
    p = 1000
    n = 100
    s = 0
    gp.ggs2dup_smart(prefix, p, n, s)
    print('Passed!')


def test_getCols():
    print('Testing getCols...')
    inFile = 'test.small.mat'
    outFile = 'test.small.sub.mat'
    othFile = 'test.small.oth.mat'
    outCorrFile = 'test.small.sub.corr.mat'
    othCorrFile = 'test.small.oth.corr.mat'
    a = 4
    X = np.arange(a * a).reshape((a, a))
    idxs = range(a)[::2]
    idxsOth = range(a)[1::2]
    Y = X[:, idxs]
    YOth = X[:, idxsOth]
    np.savetxt(inFile, X, fmt='%d')
    np.savetxt(outCorrFile, Y, fmt='%d')
    np.savetxt(othCorrFile, YOth, fmt='%d')
    pd.getCols(inFile, idxs, outFile, othFile)
    Z = np.loadtxt(outFile, dtype=np.int8)
    ZOth = np.loadtxt(othFile, dtype=np.int8)
    assert(np.all(Z == Y))
    assert(np.all(ZOth == YOth))
    print('Passed!')


def test_getEvenCols():
    print('Testing getEvenCols...')
    inFile = 'test.small.mat'
    outFile = 'test.small.sub.mat'
    othFile = 'test.small.oth.mat'
    outCorrFile = 'test.small.sub.corr.mat'
    othCorrFile = 'test.small.oth.corr.mat'
    a = 4
    X = np.arange(a * a).reshape((a, a))
    idxs = range(a)[::2]
    idxsOth = range(a)[1::2]
    Y = X[:, idxs]
    YOth = X[:, idxsOth]
    np.savetxt(inFile, X, fmt='%d')
    np.savetxt(outCorrFile, Y, fmt='%d')
    np.savetxt(othCorrFile, YOth, fmt='%d')
    pd.getEvenCols(inFile, outFile, othFile)
    Z = np.loadtxt(outFile, dtype=np.int8)
    ZOth = np.loadtxt(othFile, dtype=np.int8)
    assert(np.all(Z == Y))
    assert(np.all(ZOth == YOth))
    print('Passed!')


def test_getEvenCols_smart():
    print('Testing getEvenCols_smart...')
    gp.ggs_smart(PREFIX, P, NPLUSM)
    gp.ggs2dup_smart(PREFIX, P, NPLUSM)
    gp.getEvenCols_smart(PREFIX, P, NPLUSM)
    print('Passed!')


def test_genGenoPair():
    print('Testing genGenoPair...')
    gp.genGenoPair(PREFIX, P, N)
    print('Passed!')


def test_classic_pca_smart():
    print('Testing classic_pca_smart...')

    # # Generate the genotype
    # ref_geno, stu_geno = gp.genGenoPair(PREFIX, P, N)

    # # Convert ggs genotype file to trace genotype file
    # ref_trace_geno = gp.genname('trace_geno', PREFIX, P, N, s=0)
    # stu_trace_geno = gp.genname('trace_geno', PREFIX, P, N, s=1)
    # gp.geno2trace(ref_geno, ref_trace_geno)
    # gp.geno2trace(stu_geno, stu_trace_geno)

    # # Run trace
    # trace_ref_output, trace_stu_output = gp.trace(PREFIX, P, N)

    # Run classic
    classic_ref_X, classic_stu_X, proj_stu_X = gp.classic_pca_smart(
        PREFIX, P, N, K)

    # Load outputs
    # classic_ref_X = np.loadtxt(classic_ref_coord,
    #                            dtype=FLOAT_TYPE)
    # classic_stu_X = np.loadtxt(classic_stu_coord,
    #                            dtype=FLOAT_TYPE)
    # trace_ref_X = gp.load_trace_output_smart(PREFIX, P, N, 'ref', K)
    # trace_stu_X = gp.load_trace_output_smart(PREFIX, P, N, 'stu', K)

    # Plot the outputs
    alpha = 0.3
    plt.scatter(
        classic_ref_X[:, 0], classic_ref_X[:, 1],
        marker='s', alpha=alpha, label='classic ref'
    )
    plt.scatter(
        classic_stu_X[:, 0], classic_stu_X[:, 1],
        alpha=alpha, label='classic stu')
    # plt.scatter(
    #     proj_stu_X[:, 0], proj_stu_X[:, 1],
    #     alpha=alpha, label='proj stu')
    # plt.scatter(trace_ref_X[:, 0], trace_ref_X[:, 1], marker='s', alpha=alpha, label='trace ref')
    # plt.scatter(trace_stu_X[:, 0], trace_stu_X[:, 1], alpha=alpha, label='trace stu')

    # # Comparison
    # plt.scatter(classic_ref_X[:, 0], trace_ref_X[:, 0],
    #             label='ref PC0', alpha=alpha, marker='o')
    # plt.scatter(classic_ref_X[:, 1], trace_ref_X[:, 1],
    #             label='ref PC1', alpha=alpha, marker='o')
    # plt.scatter(classic_stu_X[:, 0], trace_stu_X[:, 0],
    #             label='study PC0', alpha=alpha, marker='o')
    # plt.scatter(classic_stu_X[:, 1], trace_stu_X[:, 1],
    #             label='study PC1', alpha=alpha, marker='o')
    # plt.xlabel('TRACE')
    # plt.ylabel('Python rewriting')

    plt.suptitle('ggs, p=' + str(P) + ', n=' + str(N))
    plt.legend()
    plt.show()

    print('Passed!')


def test_genname():
    print('Testing genname...')
    meth = 'classic'
    filename = gp.genname('coord', PREFIX, P, N, meth=meth)
    filename_corr = op.realpath(
        DATA_DIR) + '/' + PREFIX + '_' + str(P) + '_' + str(N) + '_' + meth + '.coord'
    assert(filename == filename_corr)
    print('Passed!')


def test_geno2trace():
    print('Testing geno2trace...')
    ref_geno, stu_geno = gp.genGenoPair(PREFIX, P, N)
    ref_trace = gp.genname('trace_geno', PREFIX, P, N, s=0)
    stu_trace = gp.genname('trace_geno', PREFIX, P, N, s=1)
    gp.geno2trace(ref_geno, ref_trace)
    gp.geno2trace(stu_geno, stu_trace)
    traceConfFile = gp.genTraceConf(PREFIX, P, N)
    sp.call([TRACE_EXE, '-p', traceConfFile])
    print('Check output')


def test_loadtrace():
    print('Testing load_trace_output_smart...')
    trace_ref_X = gp.load_trace_output_smart(PREFIX, P, N, 'ref', K)
    trace_stu_X = gp.load_trace_output_smart(PREFIX, P, N, 'stu', K)
    print(trace_ref_X[:5, :5])
    print(trace_stu_X[:5, :5])
    print('Passed!')


def test_procrustes():
    print('Testing procrustes...')
    X = np.array([0, 1, 2, 3, 4, 5], dtype=np.float64).reshape((3, 2))
    Y = np.array([14, 16, 10, 12, 18, 20], dtype=np.float64).reshape((3, 2))
    gp.procrustes(X, Y)
    print('Passed!')


def test_getWeightedIdxs():
    print('Testing getWeightedIdxs...')
    # n = 10
    # weight = np.array([0.3, 0.5])
    # idxs_corr = [0, 5, 6]
    # idxs = pd.getWeightedIdxs(n, weight)
    # assert(all(idxs == idxs_corr))
    n = 400
    weight = [1, 0.2, 0.2, 0.2]
    idxs = pd.getWeightedIdxs(n, weight)
    print(idxs)
    print('Passed!')


def test_getWeightedCols():
    print('Testing getWeightedCols...')
    inFile = 'test.mat'
    outFile = 'test.out.mat'
    othFile = 'test.oth.mat'
    weight = np.array([1, 2 / 3, 0.5])
    pd.getWeightedCols(inFile, outFile, othFile, weight)
    print('Passed!')


def test_ggs2trace():
    print('Testing ggs2trace...')
    ggsFile = '../data/testBlock/testBlock_1000_200_2_100.ggs'
    genoFile = '../data/testBlock/testBlock_1000_200_2_1_100.geno'
    dupFile = '../data/testBlock/testBlock_1000_200_2_1_100.geno.tr'
    pd.ggs2trace(ggsFile, genoFile)
    print('Passed!')


def test_deconcat():
    print('Testing deconcat...')
    inFile = 'test.mat.long'
    outFile = 'test.mat.long.sub'
    p = 3
    pd.deconcat(inFile, outFile, p)
    print('Check output files.')


def test_transpose():
    print('Testing transpose...')
    inFile = 'test.mat'
    outFile = 'test.mat.transposed'
    corrFile = 'test.mat.tr'
    pd.transpose(inFile, outFile)
    match = filecmp.cmp(outFile, corrFile, shallow=False)
    assert(match)
    print('Passed!')


def test_transpose_multi():
    print('Testing transpose_multi...')
    inSuff = 'test.mat.long.sub'
    outSuff = 'test.mat.long.sub.tr'
    n = 2
    pd.transpose_multi(inSuff, outSuff, n)
    print('Check output')


def test_transpose_block():
    print('Testing transpose_block...')
    print('int matrix...')
    inFile = 'test.mat.large'
    outFile = 'test.mat.large.tr'
    outCorrFile = 'test.mat.large.tr.corr'
    X = np.arange(15).reshape((5, 3))
    np.savetxt(inFile, X, fmt=INT_FMT, delimiter=DELIM)
    np.savetxt(outCorrFile, X.transpose(), fmt=INT_FMT, delimiter=DELIM)
    p = 3
    q = 2
    pd.transpose_block(inFile, outFile, p, q)
    match = filecmp.cmp(outFile, outCorrFile, shallow=False)
    assert(match)
    print('Passed!')
    print('char matrix...')
    # Now transpose a char matrix
    inFile = 'mat.char'
    outFile = 'mat.char.tr'
    outCorrFile = 'corr.mat.char.tr'
    pd.transpose_block(inFile, outFile, p, q)
    match = filecmp.cmp(outFile, outCorrFile, shallow=False)
    assert(match)
    print('Passed!')
    print('geno.tr file...')
    direc = '/home/david/data/ggsim/'
    inFile = direc + 'ggsim_4_4_2_1_100.geno.tr'
    outFile = direc + 'ggsim_4_4_2_1_100.geno.tr.tr'
    outCorrFile = direc + 'ggsim_4_4_2_1_100.geno'
    p = 4
    q = 3
    pd.transpose_block(inFile, outFile, p, q)
    match = filecmp.cmp(outFile, outCorrFile, shallow=False)
    assert(match)
    print('Passed!')
    print('Passed!')


def test_concat_col():
    print('Testing concat_col...')
    inSuff = 'test.mat.long.sub.tr'
    outSuff = 'test.mat.long.sub.merged'
    M = 2
    N = 5
    q = 2
    inFiles = []
    for i in range(q):
        inFile = pd.suff2file(inSuff, i)
        inFiles += [inFile]
    pd.concat_col(inFiles, outSuff, M, N, q)
    print('Check output')


def test_concat():
    print('Testing concat...')
    inSuff = 'test.mat.large.tr_a'
    outFile = 'test.mat.large.tr'
    inFiles = []
    for i in range(2):
        inFile = pd.suff2file(inSuff, i)
        inFiles += [inFile]
    pd.concat(inFiles, outFile)
    print('Check output')


def test_saveMat():
    print('Test saveMat...')
    inFile = 'mat.char'
    outFile = 'mat.char.save'
    X = [line.split() for line in open(inFile)]
    pd.saveMat(outFile, X)
    match = filecmp.cmp(outFile, inFile, shallow=False)
    assert(match)
    print('Passed')


def test_all():
    print('Start unit testing for genoPCA...')
    # test_genname()
    # test_ggs()
    # test_ggs_smart()
    # test_ggs2dup()
    # test_ggs2dup_smart()
    # test_getCols()
    # test_getEvenCols()
    # test_getEvenCols_smart()
    # test_genGenoPair()
    # test_geno2trace()
    # test_loadtrace()
    # test_classic_pca_smart()
    # test_procrustes()
    # test_getWeightedIdxs()
    # test_getWeightedCols()
    test_ggs2trace()
    # test_transpose()
    # test_deconcat()
    # test_transpose_multi()
    # test_concat_col()
    # test_saveMat()
    # test_transpose_block()
    # test_concat()
    print('All tests passed!')


test_all()
