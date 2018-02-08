import numpy as np
import subprocess
import math

DELIM = '\t'
P = 1000
Q = 50000


def getNCols(fileName):
    ''' find the number of columns in a file '''
    with open(fileName, 'r') as text:
        line = text.readline()
        n = len(str(line).split())
    return(n)


def getRows(inFile, rowIdxs, outFile, otherFile):
    '''Get rows from inFile by rowIdxs as index and save in outFile'''
    print("Getting rows from index...")
    with open(inFile, 'r') as inf, open(outFile, 'w') as outf, open(otherFile, 'w') as othf:
        for i, line in enumerate(inf):
            if (i in rowIdxs):
                outf.write(line)
            else:
                othf.write(line)
            if i % 100000 == 0:
                print('Finished line ' + str(i))
    print("Done!")


def getCols(inFile, idxs, outFile, otherFile):
    with open(inFile, 'r') as inf, open(outFile, 'w') as outf, open(otherFile, 'w') as othf:
        for i, line in enumerate(inf):
            lst = str(line).split()
            n = len(lst)
            outlst = [lst[i] for i in idxs]
            othlst = [lst[i] for i in range(n) if i not in idxs]
            outst = '\t'.join(outlst) + '\n'
            othst = '\t'.join(othlst) + '\n'
            outf.write(outst)
            othf.write(othst)


def getRandRows(inFile, n, nn, outFile, otherFile):
    print("Getting " + str(n) + " random rows out of " + str(nn) + " rows ...")
    rowIdxs = np.random.choice(nn, size=n, replace=False)
    getRows(inFile, rowIdxs, outFile, otherFile)


def getEvenRows(inFile, nn, outFile, otherFile):
    print("Getting every other row of " + str(nn) + " rows ...")
    n = nn / 2
    rowIdxs = list(range(n))
    rowIdxs = [i * 2 for i in rowIdxs]
    getRows(inFile, rowIdxs, outFile, otherFile)


def getEqSpacedRows(inFile, nplusm, n, outFile, otherFile):
    print('Getting equally spaced ' + str(n) +
          ' rows out of ' + str(nplusm) + 'rows ...')
    k = nplusm // n
    rowIdxs = list(range(n))
    rowIdxs = [i * k for i in rowIdxs]
    print(rowIdxs)
    getRows(inFile, rowIdxs, outFile, otherFile)


def getEvenCols(inFile, outFile, othFile):
    n = getNCols(inFile)
    idxs = range(n)[::2]
    getCols(inFile, idxs, outFile, othFile)


def ggs2trace(ggsFile, genoFile, weight=[1]):
    ''' Convert a genotype simulated by ggs to TRACE's .geno format. Every two haploids are added to form a duploid.'''
    print("Converting ggs output")
    print(ggsFile)
    print("to trace input")
    print(genoFile)
    print("with weight")
    print(weight)
    print("...")
    print("Creating duploids...")
    nCols = getNCols(ggsFile) - 1  # The first column is the row names
    if nCols % 2 != 0:
        print(nCols)
        raise Exception('The number of columns must be even.')
    dupFile = genoFile + ".tr"
    idxs = getWeightedIdxs(nCols, weight)
    outNCols = len(idxs) // 2
    with open(ggsFile, 'r') as ggsF, open(dupFile, 'w') as dupF:
        # st = '\t'.join(['indv'] * (outNCols)) + '\n'
        # dupF.write(st)
        # st = '\t'.join(['popu'] * (outNCols)) + '\n'
        # dupF.write(st)
        header = ggsF.readline()

        # make fun: split_header(header), output indv, popu
        header_lst = header.split('\t')
        # Remove the trash at beginning and end
        header_lst = header_lst[1:-1]
        # Creating header for duploids
        header_lst = header_lst[::2]
        header_mat = [elem.split('/') for elem in header_lst]
        popu_info = ['/'.join(elem[0:2]) for elem in header_mat]
        indv_info = header_lst
        popu_info_str = '\t'.join(popu_info) + '\n'
        indv_info_str = '\t'.join(indv_info) + '\n'
        dupF.write(popu_info_str)
        dupF.write(indv_info_str)
        # end fun

        for i, line in enumerate(ggsF):
            lst = line.split()
            lst = list(map(int, lst[1:]))  # skip the row name
            lst = [lst[i] for i in idxs]
            lst = [x + y for x, y in zip(lst[0::2], lst[1::2])]
            lst = list(map(str, lst))
            st = '\t'.join(lst) + '\n'
            dupF.write(st)
            if i % 100000 == 0:
                print("Finished line " + str(i))
    print("Finished!")
    print("Duploid file written to " + dupFile)
    print("Transposing " + dupFile + " into " + genoFile + "...")
    transpose_block(dupFile, genoFile, p=P, q=Q)
    # subprocess.call(["./transpose", dupFile, genoFile])
    # subprocess.call(["rm", dupFile])
    print("Finished!")
    print("Geno file written to " + genoFile)


def ggsChild2trace(ggsFile, genoFile, nChild):
    '''
    Simulate children's genotypes from gynotypes simulated by ggs and convert the children to TRACE's .geno format.
    Every two haploids are combined to form a duploid.
    A child is simulated from two adjacent individuals by treating each geneaology as independent with a 0.5 probability of being selected.
    nChild many children are genrated from every parents.
    '''
    print("Generating child duploids...")
    nCols = getNCols(ggsFile) - 1  # The first column is the row names
    if nCols % 4 != 0:
        raise Exception('The number of columns must be a multiple of 4.')
    dupFile = genoFile + ".tr"
    with open(ggsFile, 'r') as ggsF, open(dupFile, 'w') as dupF:
        st = '\t'.join(['indv'] * (nCols // 4 * nChild)) + '\n'
        dupF.write(st)
        st = '\t'.join(['popu'] * (nCols // 4 * nChild)) + '\n'
        dupF.write(st)
        next(ggsF)  # skip the header
        hapChoice = []
        for i, line in enumerate(ggsF):
            lst = line.split()
            # print(lst)
            rowname = lst[0]
            rowname = rowname.split(',')
            loci = rowname[1]
            if loci == '0':  # the start of a new geneaology
                # select which haploid to use
                hapChoice = np.random.randint(
                    low=0, high=2, size=nCols // 2 * nChild)
                coupleIdx = range(nCols // 4)
                indvIdx = [[i * 2, i * 2 + 1] for i in coupleIdx]
                indvMultIdx = np.repeat(indvIdx, nChild, axis=0)
                hapMultIdx = indvMultIdx * 2
                hapMultIdx = np.ndarray.flatten(hapMultIdx)
                hapChoice += hapMultIdx
            # print(hapChoice)
            lst = list(map(int, lst[1:]))
            lst = [lst[k] for k in hapChoice]
            # print(lst)
            lst = [x + y for x, y in zip(lst[0::2], lst[1::2])]
            lst = list(map(str, lst))
            # print(lst)
            st = '\t'.join(lst) + '\n'
            dupF.write(st)
            if i % 100000 == 0:
                print("Finished line " + str(i))
    print("Finished!")
    print("Transposing file...")
    subprocess.call(["./transpose", dupFile, genoFile])
    subprocess.call(["rm", dupFile])
    print("Finished!")


def getWeightedCols(inFile, outFile, othFile, weight):
    n = getNCols(inFile)
    idxs = getWeightedIdxs(n, weight)
    getCols(inFile, idxs, outFile, othFile)


def getWeightedIdxs(n, weight):
    weight = np.array(weight)
    kk = len(weight)
    assert(np.all(weight >= 0))
    assert(np.all(weight <= 1))
    l = n // kk
    weightL = weight * l
    weightL = np.around(weightL)
    weightL = np.array(weightL, dtype=np.int_)
    idxs = np.array([], dtype=int)
    for i in range(kk):
        startI = i * l
        nI = weightL[i]
        stopI = startI + nI
        idxsI = np.arange(startI, stopI)
        idxs = np.concatenate((idxs, idxsI))
    return idxs


def transpose(inFile, outFile, delim='\t'):
    X = [line.split() for line in open(inFile, 'r')]
    X_tr = list(map(list, zip(*X)))
    saveMat(outFile, X_tr, delim)
    # If we want to use numpy:
    # X = np.loadtxt(inFile, dtype=INT_TYPE)
    # X = np.transpose(X)
    # np.savetxt(outFile, X, fmt=INT_FMT, delimiter=delim)


def transpose_multi(inSuff, outSuff, n, delim='\t'):
    for i in range(n):
        inFile = suff2file(inSuff, i)
        outFile = suff2file(outSuff, i)
        transpose(inFile, outFile, delim)


def deconcat(inFile, outSuff, p):
    n = 0
    outF = None
    with open(inFile, 'r') as inF:
        for i, line in enumerate(inF):
            if i % p == 0:
                if outF is not None:
                    outF.close()
                    n += 1
                outFile = outSuff + '_' + str(n)
                open(outFile, 'w').close()  # Clear content
                outF = open(outFile, 'a')
            outF.write(line)


def concat_col(inFiles, outFile, M, N, q, delim='\t'):
    # Form a M * N matrix by merging q blocks by columns
    # No sanity check for whether dimensions match
    # X = np.zeros((M, N), dtype=INT_TYPE)
    # X = [[None] * N] * M
    X = [[0 for j in range(N)] for i in range(M)]
    start = 0
    for i in range(q):
        inFile = inFiles[i]
        # incre = pd.getNCols(inFile)
        # end = start + incre
        # assert(end <= N)
        # X[:, start:end] = np.loadtxt(inFile, dtype=INT_TYPE)
        X_sub = [line.split() for line in open(inFile, 'r')]
        for j in range(M):
            for k in range(len(X_sub[0])):
                X[j][start + k] = X_sub[j][k]
        start += len(X_sub[0])
    # np.savetxt(outFile, X, fmt=INT_FMT, delimiter=delim)
    saveMat(outFile, X, delim=delim)


def transpose_block(inFile, outFile, p, q):
    M = getNRows(inFile)
    N = getNCols(inFile)
    m = int(math.ceil(M / p))
    n = int(math.ceil(N / q))
    deconcat(inFile, inFile, p)
    transpose_multi(inFile, outFile, m)
    for i in range(m):
        outFileCol = suff2file(outFile, i)
        deconcat(outFileCol, outFileCol, q)
    outRows = []
    for j in range(n):
        outRowBlocks = []
        for i in range(m):
            outBlock = suff2file(suff2file(outFile, i), j)
            outRowBlocks += [outBlock]
        outRow = suff2file(outFile + "_a", j)
        outRows += [outRow]
        qq = q
        if j == n - 1:
            qq = N % q
        concat_col(outRowBlocks, outRow, qq, M, m)
    concat(outRows, outFile)
    subprocess.call(["rm", inFile + "_*"])
    subprocess.call(["rm", outFile + "_*"])


def concat(inFiles, outFile):
    # Concatenate files by rows
    with open(outFile, 'w') as outF:
        for inFile in inFiles:
            with open(inFile) as inF:
                outF.write(inF.read())


def suff2file(suff, n):
    filename = suff + '_' + str(n)
    return filename


def saveMat(outFile, X, delim=DELIM):
    open(outFile, 'w').close()
    with open(outFile, 'a') as outF:
        for lst in X:
            line = delim.join(lst)
            outF.write(line + '\n')


def getNRows(fname):
    i = -1
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1
