import frugalpca_tests as fpt
import pandas as pd
import numpy as np
import subprocess
import multiprocessing as mp
NUM_CORES = mp.cpu_count()

print('Number of cores: ' + str(NUM_CORES))

# fpt.convert()
# fpt.test_pca_subpopu(pref_ref, pref_stu, 'EUR', cmp_trace=True)

# fpt.convert_ggsim(4)
# for i in range(5):
#     fpt.test_pca_ggsim(i)
# fpt.test_split_bed_indiv()

# fpt.test_pca_ggsim()
# fpt.test_pca_5c()
# fpt.test_merge_array_results()

# fpt.add_pure_stu()
fpt.test_pca_5c_EUR()
# fpt.test_pca_EUR_pure()
# fpt.test_pca_5c_EUR_impure()
# fpt.test_pca_5c_EUR_impure_predrefpopu()
# fpt.test_pca_EUR_pure_homogen()
# fpt.test_pca_5c_EUR_impure_ref_homogen()
