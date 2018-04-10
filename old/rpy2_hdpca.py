import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
import numpy as np

# This is the R instance
r = robjects.r

# Activate conversion from numpy to R
rpy2.robjects.numpy2ri.activate()

# Load hdpca package
importr('hdpca')

# Load the hdpca example dataset into R
r('data(Example)')

# Create rpy2 object (function) that correspond to the function in R
pc_adjust = r['pc_adjust']

# Convert vectors in R into numpy arrays
train_eval = np.array(r['train.eval'])
test_scores = np.array(r['testscore'])
p = int(r['p'][0])
n = int(r['n'][0])

# Run the hdpca method
# Remember to translate . into _ and FALSE into False
out = pc_adjust(train_eval, p, n, test_scores, method="d.gsp", n_spikes_max=20)
out_np = np.array(out)
print(out_np[:4])

print("Done")
