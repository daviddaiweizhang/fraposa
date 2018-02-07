import procdata as pd
import numpy as np
import sys

inFile = sys.argv[1]
outFile = sys.argv[2]
othFile = sys.argv[3]
s = int(sys.argv[4])
weight = np.array([1.0, 1.0 / s, 1.0 / s, 1.0 / s])
pd.getWeightedCols(inFile, outFile, othFile, weight)
