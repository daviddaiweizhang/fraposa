import procdata as pd
import sys

inFile = sys.argv[1]
n = int(sys.argv[2])
nplusm = int(sys.argv[3])
outFile = sys.argv[4]
otherFile = sys.argv[5]
# pd.getEqSpacedRows(inFile, nplusm, n, outFile, otherFile)
# pd.getEvenRows(inFile, nplusm, outFile, otherFile)
pd.getRandRows(inFile, n, nplusm, outFile, otherFile)
