import procdata as pd
import sys

pref = sys.argv[1]
p = int(sys.argv[2])
n = int(sys.argv[3])
k = int(sys.argv[4])
s = int(sys.argv[5])
mig = int(sys.argv[6])

# print('Using testing parameters for ggs2trace')
# pref = '../data/ggsim/ggsim'
# p = 1000
# n = 100 + 100
# s = 1
# h = 2 * 2

h = k * k
n_reduced = n // h + n // (h * s) * (h - 1)
ggsFile = pref + '_' + str(p) + '_' + str(n) + '_' + \
    str(k) + '_' + str(mig) + '.ggs'
genoFile = pref + '_' + str(p) + '_' + str(n_reduced) + \
    '_' + str(k) + '_' + str(s) + '_' + str(mig) + '.geno'
weight = [1.0] + [1.0 / s] * (h - 1)
# Something went wrong with weight after transpose_block is used
# Temporarily disabled weight option
# pd.ggs2trace(ggsFile, genoFile, weight)
pd.ggs2trace(ggsFile, genoFile)
