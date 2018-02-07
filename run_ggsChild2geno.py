import procdata as pd
import sys

fp = sys.argv[1]
n = int(sys.argv[2])
m = int(sys.argv[3])
nChild = m // (n // 2)
ggsFile = fp + "_" + str(n) + ".ggs"
genoFile = fp + "_" + str(m) + ".geno"
pd.ggsChild2geno(ggsFile, genoFile, nChild)

