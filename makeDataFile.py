
# def normArray(x):
#     y = np.array(x)
#     n = len(y)
#     xbar = np.mean(y)
#     xdev = np.std(y) 
#     y = y - xbar
#     if xdev > 0:
#         y = y / xdev
#     return(y)

# chromoInfoFile = '../data/HGDP_Map'
# chromoInfoFileCol = 1
# chromoInfo = getChromoInfo(chromoInfoFile, chromoInfoFileCol)


# chromoNum = "21"
# subsetSNPsFile = snpsFile + "_chromo" + chromoNum
# getSNPsOnChromo(chromoNum, snpsFile, chromoInfo, subsetSNPsFile)

# popFile = "../data/hgdp_population"
# popFileCol = 3
# testIdx, trainIdx = getUnqPops(popFile, popFileCol)



# pIdx = np.random.choice(pAll, p, replace=False) 
# nIdx = np.random.choice(nAll, n, replace=False)

# idxConvertFile = "../data/idxConvert"
# toIdx2 = np.loadtxt(idxConvertFile, dtype="int")
# toIdx2 = toIdx2 - 1 # start counting by zero
# nIdx2 = toIdx2[nIdx]

# def getTrainTest(allFile, trainIdx, testIdx, trainFile, testFile):

#     print("Dividing matrix into training and testing matrices by column...")
#     # clear the output file
#     open(trainFile, 'w').close()
#     open(testFile, 'w').close()

#     counter = 0
#     with open(allFile, "r") as all, open(trainFile, "ab") as train, open(testFile, "ab") as test:
#         for line in all:
#             x = strToArray(line)
#             x = np.array(x)
#             xTest = x[testIdx]
#             xTest = np.matrix(xTest)
#             xTrain = x[trainIdx]
#             xTrain = np.matrix(xTrain)
#             np.savetxt(train, xTrain, fmt='%10.5f')
#             np.savetxt(test, xTest, fmt='%10.5f')

#             counter += 1
#             if counter % 10000 == 0:
#                 print("Finished processing line " + str(counter))

#     print("Done!")



# def getTrainTestLargeMemUse(allFile, trainIdx, testIdx, trainFile, testFile):

#     print("Dividing matrix into training and testing matrices...")

#     # read the SNP file(p * n) for individuals
#     # divide the individuals into a training and a testing group, based ontrainIdx and testIdx
#     # save the training group and testing group into text files

#     print("Loading allFile " + allFile + "... ")
#     xAll = np.loadtxt(allFile)
#     xAll = np.matrix(xAll)
#     print("Done!")
#     print("Create testing matrix...")
#     xTest = xAll[:, testIdx]
#     print("Done!")
#     print("Creating training matrix...")
#     xTrain = xAll[:, trainIdx]
#     print("Done!")

#     print("Saving training matrix...")
#     np.savetxt(trainFile, xTrain, fmt='%10.5f')
#     print("Done!")
#     print("Saving testing matrix...")
#     np.savetxt(testFile, xTest, fmt='%10.5f')
#     print("Done!")

#     p = np.shape(xTrain)[0]
#     nTest = np.shape(xTest)[1]
#     nTrain = np.shape(xTrain)[1]
#     nAll = nTest + nTrain

#     print("There are " + str(p) + " SNPs.")
#     print("There are " + str(nTest) + " individuals in the testing group.")
#     print("There are " + str(nTrain) + " individuals in the training group.")



import numpy as np
import pandas as pd
from shutil import copyfile
import os



# convert a string to a list of floats
def strToArray(s):
    stringList = s.split()
    n = len(stringList)
    numList = [None]*n
    for i,st in enumerate(stringList):
        if(st != "NA"):
            numList[i] = int(st)
    return (numList)

def cleanList(x, good, missing):
    for i,w in enumerate(x):
        if w not in good:
            x[i] = missing
            
    # y = [z for z in x if z in good]
    # n = len(y)
    # if n == 0:
    #     return([missing] * len(x))
    # ybar = sum(y) / (n * 1.0)
    # ybar = int(round(ybar))
    # for i,w in enumerate(x):
    #     if w is None:
    #         x[i] = ybar
    # return(x)

def cleanFile(inFile, outFile, good, missing):
    print("Replacing unwanted values with " + missing + "...")
    open(outFile, 'w').close()
    with open(inFile, "r") as inf, open(outFile, "w") as outf:
        counter = 0
        for line in inf:
            l = line.split()
            cleanList(l, good, missing)
            for w in l:
                outf.write(str(w) + "\t")
            outf.write("\n")
            counter = counter + 1
            if counter % 10000 == 0:
                print("Finished processing line " + str(counter))
    print("Done!")

            

def normFile(inFile, outFile, missing, mean = 0, sd = 0):
    print("Normalizing " + inFile + " and replacing " + missing + " with row mean...")
    meanSdFile = outFile + "meanSd"
    open(outFile, 'w').close()
    open(meanSdFile, 'w').close()
    with open(inFile, "r") as inf, open(outFile, "ab") as outf, open(meanSdFile, "ab") as mnsdf:
        for i,line in enumerate(inf):
            l = line.split()
            x = [float(w) for w in l if (w != missing)]

            if mean == 0: # no input value for mean
                if not x:
                    mean = 0
                else:
                    mean = np.mean(x)

            l = [float(w) if (w != missing) else mean for w in l]
            l = np.array([l])
            l = l - mean
            if sd == 0:
                sd = np.std(l, ddof = 1) 
            if(sd > 0):
                l = l / sd
            elif sd < 0:
                print("Standard deviation cannot be negative")
                break
            mnsd = np.matrix([[mean, sd]])
            np.savetxt(outf, l, fmt = "%1.4f", delimiter = "\t")
            np.savetxt(mnsdf, mnsd, fmt = "%1.4f", delimiter = "\t")
            if i % 10000 == 0:
                print("Finished processing line " + str(i))
    print("Done!")
        
    
# get column k from file


def getCol(fileName, k):
    col = []
    with open(fileName) as file:
        for line in file:
            arr = line.split()
            if len(arr) > k:
                thisElem = arr[k]
                col.append(thisElem)
    return (col)


# find the number of rows in a file
def getNRow(filename):
    with open(filename) as f:
        return( sum(1 for _ in f))


def getChromoInfo(chromoInfoFile, chromoCol):
    # gets the chromosome locations of the SNPs and saves them into an array
    print("Reading chromosome info...")
    chromoInfo = getCol(chromoInfoFile, chromoCol)
    print("Finished reading the chromosome locations of " +
          str(len(chromoInfo)) + " SNPs")
    return(chromoInfo)


def getSNPsOnChromo(chromoNum, snpsFile, chromoInfo, outFileName):
    # read the SNP file
    # use the chromosone location array to select the SNPs on the chromosome wanted
    # save the selected SNPs into a file

    # find the number of rows
    p = chromoInfo.count(chromoNum)

    # find the number of columns
    n = getNCol(snpsFile)
    with open(snpsFile, 'r') as text:
        line = text.readline()
        n = len(str(line).split())

    xAll = np.zeros((p, n))

    # read the data
    print("Reading SNPs...")
    j = 0
    with open(snpsFile, 'r') as snps:
        for i, line in enumerate(snps):
            if chromoInfo[i] == chromoNum:
                arr = strToArray(line)
                xAll[j, :] = arr
                j = j + 1
            if i % 1e5 == 0:
                print("Processed " + str(i) + " SNPs.")

    # writing result to file
    open(outFileName, 'w').close()
    np.savetxt(outFileName, xAll, fmt='%10.5f')

    print("Finished checking " + str(len(chromoInfo)) + " SNPs.")
    print("There are " + str(np.shape(xAll)
                             [0]) + " SNPs on chromosome " + str(chromoNum) + ".")


def getUnqPops(popFile, popCol):

    # read the population info for individuals
    # return an integer array (unqIdx): the index for the first individuals in the populations
    # return a boolean array (isUnq): True = is the first individualin the
    # population, False = otherwise

    # all the populations
    pops = getCol(popFile, popCol)
    del pops[0]  # delete the header
    pops = np.array(pops)
    nPops = len(pops)

    # the unique populations
    unqPops, unqIdx = np.unique(pops, return_index=True)

    # the complement of the unique populations
    unqIdxComp = np.delete(np.arange(nPops), unqIdx, 0)

    # # a boolean array for uniqueness
    # isUnq = np.zeros(nPops, dtype=bool)
    # isUnq[unqIdx] = 1

    nUnq = unqPops.size
    print("There are " + str(nUnq) + " populations in file.")

    return unqIdx, unqIdxComp


def dvdFileByRows(inFile, rowIdx, outFile0, outFile1, both):
    print("Dividing file by rows...")
    # clear the output file
    open(outFile0, 'w').close()
    if(both):
        open(outFile1, 'w').close()
    counter = 0
    with open(inFile, "r") as inf, open(outFile0, "ab") as of0, open(outFile1, "ab") as of1:
        for line in inf:
            if counter < n: 
                of1.write(line)
            elif not(both):
                break
            else:
                of0.write(line)
            counter += 1
            if counter % 10000 == 0:
                print("Finished processing line " + str(counter))
    print("Done!")


def midRows(inFile, outFile, n1, n2, skip = 1, complement = False):
    print("Getting every " + str(skip) + " rows from row " + str(n1) + " to row " + str(n2))
    print("Input file name: " + inFile)
    print("Output file name: " + outFile)
    # clear the output file
    open(outFile, 'w').close()
    counter = 0
    with open(inFile, "r") as inf,  open(outFile, "ab") as outf:
        for line in inf:
            if counter >= n1 and counter < n2 and (counter - n1) % skip == 0:
                if counter < n2:
                    if (counter - n1) % skip == 0:
                        outf.write(line)
                else:
                    break
            counter += 1
            if counter % 10000 == 0:
                print("Finished processing line " + str(counter))
    print("Done!")


def tailRows(inFile, n, outFile):
    print("Getting the last " + str(n) + " rows of file...")
    # clear the output file
    open(outFile, 'w').close()
    counter = 0
    with open(inFile, "r") as inf,  open(outFile, "ab") as outf:
        for line in inf:
            if counter >= n:
                of1.write(line)
            counter += 1
            if counter % 10000 == 0:
                print("Finished processing line " + str(counter))
    print("Done!")



def tailRows(inFile, n, outFile0, outFile1, both):
    print("Getting the last " + str(n) + " rows of file...")
    # clear the output file
    open(outFile0, 'w').close()
    if(both):
        open(outFile1, 'w').close()
    counter = 0
    with open(inFile, "r") as inf, open(outFile0, "ab") as of0, open(outFile1, "ab") as of1:
        for line in inf:
            if counter >= n:
                of1.write(line)
            elif(both):
                of0.write(line)
            counter += 1
            if counter % 10000 == 0:
                print("Finished processing line " + str(counter))
    print("Done!")





def dvdFileByCols(inFile, colIdx1, outFile0, outFile1, both):

    print("Dividing file by columns...")
    # clear the output file
    open(outFile0, 'w').close()
    if(both):
        open(outFile1, 'w').close()
    n = getNCol(inFile)
    colIdx0 = list(set(range(n)) - set(colIdx1))

    counter = 0
    with open(inFile, "r") as inf, open(outFile0, "ab") as of0, open(outFile1, "ab") as of1:
        for line in inf:
            x = line.split()

            if(both):
                x0 = [x[i] for i in colIdx0]
                for e in x0:
                    of0.write(e + " ")
                of0.write("\n")

            x1 = [x[i] for i in colIdx1]
            for e in x1:
                of1.write(e + " ")
            of1.write("\n")

            counter += 1
            if counter % 10000 == 0:
                print("Finished processing line " + str(counter))
    print("Done!")


def midCols(inFile, n1, n2, outFile):
    print("Getting column " + str(n1) + " to column " + str(n2) + " of " + inFile)
    open(outFile, 'w').close()
    counter = 0
    with open(inFile, "r") as inf, open(outFile, "ab") as outf:
        for line in inf:
            x = line.split()
            xSub = x[n1:n2]
            for e in xSub:
                outf.write(e + "\t")
            outf.write("\n")
            counter += 1
            if counter % 10000 == 0:
                print("Finished processing line " + str(counter))
    print("Done!")




def headCols(inFile, n, outFile):

    print("Getting the last " + str(n) + " columns of file...")
    # clear the output file
    open(outFile, 'w').close()
    nAll = getNCol(inFile)

    counter = 0
    with open(inFile, "r") as inf, open(outFile, "ab") as outf:
        for line in inf:
            x = line.split()
            x1 = x[:n]
            for e in x1:
                outf.write(e + " ")
            outf.write("\n")

            counter += 1
            if counter % 10000 == 0:
                print("Finished processing line " + str(counter))
    print("Done!")


    
def tailCols(inFile, n, outFile):

    print("Getting the last " + str(n) + " columns of file...")
    # clear the output file
    open(outFile, 'w').close()
    nAll = getNCol(inFile)

    counter = 0
    with open(inFile, "r") as inf, open(outFile, "ab") as outf:
        for line in inf:
            x = line.split()
            x1 = x[-n:]
            for e in x1:
                outf.write(e + " ")
            outf.write("\n")

            counter += 1
            if counter % 10000 == 0:
                print("Finished processing line " + str(counter))
    print("Done!")

# print("Making data files...")

# # allFile = "../data/hgdp/hgdp_632958"
# # nAll = getNCol(allFile)
# # pAll = getNRow(allFile)

# p = 570859
# nTrain = 500
# nTest = 100


# # Trace

# # allFile2 = "../data/HGDP/HGDP_632958.geno"
# allFile2 = "../data/1000genomes/chrALL_570859.geno"
# allSiteFile2 = allFile2[:-5] + ".site"
# nAll2 = getNRow(allFile2)
# pAll2 = getNCol(allFile2)

# # # Reduce the number of SNPs
# # redPFile2 = allFile2[:-11] + str(p) + ".geno"
# # redPSiteFile2 = allSiteFile2[:-11] + str(p) + ".site"
# # # midCols(allFile2, 0, p+2, redPFile2) # First two cols are pop info
# # # midRows(allSiteFile2, redPSiteFile2, 0, p+1) # First row is header

# # Reduce the number of individuals
# redPFile2 = allFile2 # Uncomment this if not reducing p
# redPSiteFile2 = allSiteFile2 # Uncomment this if not reducing p
# redPTrainFile2 =  redPFile2[:-5] + "_mod_" + str(nTrain) + ".geno"
# redPTestFile2 =  redPFile2[:-5] + "_mod_" + str(nTest) + ".geno"
# # midRows(redPFile2, redPTrainFile2, 0, nTrain)
# sk = nAll2 / nTest
# midRows(redPFile2, redPTestFile2, 0, sk * nTest, skip = sk)

# # copyfile(redPSiteFile2, redPTrainFile2[:-5] + ".site")
# copyfile(redPSiteFile2, redPTestFile2[:-5] + ".site")





# os.system("cd ../data/laser; bash runTrace.sh " + str(p) + " " + str(nTrain) + " " + str(nTest))


# # Online 


# redPFile = allFile
# redPFile = allFile[:-6] + str(p)
# # midRows(allFile, 0, p, redPFile)

# redPTrainFile = redPFile + "_h" + str(nTrain)
# redPTestFile = redPFile + "_t" + str(nTest) 
# # midCols(redPFile, 0, nTrain, redPTrainFile)
# # midCols(redPFile, nAll-nTest, nAll, redPTestFile)

# os.system("cd ../data/HGDP; bash HGDP_to_hgdp.sh " + str(p) + " " + str(nTrain) + " " + str(nTest))

# cleanFile(redPTrainFile, redPTrainFile+"_cleaned", ["0","1","2"], "NA")
# cleanFile(redPTestFile, redPTestFile+"_cleaned", ["0","1","2"], "NA")

# normFile(redPTrainFile+"_cleaned", redPTrainFile+"_cleaned_normed", "NA")
# normFile(redPTestFile+"_cleaned", redPTestFile+"_cleaned_normed", "NA")



# os.system("Rscript --vanilla onlinePC.R " + str(p) + " " + str(nTrain) + " " + str(nTest))



# print("All finished!")
