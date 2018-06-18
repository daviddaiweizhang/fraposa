import frugalpca_tests as fpt
import multiprocessing as mp
from joblib import Parallel, delayed
NUM_CORES = mp.cpu_count()

print('Number of cores: ' + str(NUM_CORES))

# fpt.convert()
# fpt.test_pca_subpopu(pref_ref, pref_stu, 'EUR', cmp_trace=True)

# fpt.test_pca_ggsim()
# fpt.test_pca_5c()
# fpt.test_pca_5c_EUR()
# Parallel(n_jobs=NUM_CORES)(delayed(fpt.convert_ggsim)(i) for i in range(5))
# fpt.convert_ggsim(4)
# fpt.test_pca_ggsim()
fpt.test_merge_array_results()

