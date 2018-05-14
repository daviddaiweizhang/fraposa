import pandas as pd
import numpy as np
import frugalpca as fp
import subprocess


def test_online_svd_procrust():
    np.random.seed(21)
    # def test_svd_online():
    print('Testing test_svd_online...')

    # # For debugging only
    # # For comparing with the R script written by Shawn
    # X = np.loadtxt('test_X.dat')
    # b = np.loadtxt('test_b.dat').reshape((-1,1))
    # p, n = X.shape

    # Generate testing matrices
    p = 1000
    n = 200
    X = np.random.normal(size = p * n).reshape((p,n))
    b = np.random.normal(size = p).reshape((p,1))
    np.savetxt('test_X.dat', X)
    np.savetxt('test_b.dat', b)

    # Center reference data
    X_mean = np.mean(X, axis = 1).reshape((p,1))
    X -= X_mean
    # Nonrmalize referencd data
    X_norm = np.std(X, axis = 1).reshape((p,1))
    X_norm[X_norm == 0] = 1
    X /= X_norm

    # Center study data
    b -= X_mean
    b /= X_norm

    # Parameters for onlineSVD
    svd_online_dim = 100 # Number of PC's calculated by online SVD
    PC_new_dim = 20 # Number of PC's we want for each new sample
    PC_ref_dim = 4 # Number of PC's for the reference group
    assert PC_new_dim <= svd_online_dim
    assert PC_ref_dim <= PC_new_dim

    # Decompose the training matrix
    U, d, Vt = np.linalg.svd(X, full_matrices = False)
    V = Vt.transpose()
    PC_ref = V[:, :PC_ref_dim]
    # Subset the PC scores since we only need the first k PC's
    U1 = U[:, :svd_online_dim]
    d1 = d[:svd_online_dim]
    V1 = V[:, :svd_online_dim]
    d2, PC_new = fp.svd_online(U1, d1, V1, b, PC_new_dim)

    # Test if the result is close enough
    trueAns = np.linalg.svd(np.concatenate((X,b),axis=1))[2].transpose()[:,:PC_new_dim]
    for i in range(trueAns.shape[1]):
        assert \
            abs(np.max(PC_new[:,i] - trueAns[:,i])) < 0.05 or \
            abs(np.max(PC_new[:,i] + trueAns[:,i])) < 0.05 # online_svd can flip the sign of a PC
    print('Passed!')

    print('Testing procrustes...')
    PC_new_head, PC_new_tail = PC_new[:-1, :], PC_new[-1, :].reshape((1,PC_new_dim))
    PC_ref_fat = np.zeros(n * PC_new_dim).reshape((n, PC_new_dim))
    PC_ref_fat[:, :PC_ref_dim] = PC_ref
    np.savetxt('test_PC_ref.dat', PC_ref)
    np.savetxt('test_PC_ref_fat.dat', PC_ref_fat)
    np.savetxt('test_PC_new_head.dat', PC_new_head)
    # Test procrustes with the same dimension
    R, rho, c = fp.procrustes(PC_ref_fat, PC_new_head)
    # PC_new_tail_trsfed = PC_new_tail @ R * rho + c
    # PC_new_tail_trsfed = PC_new_tail_trsfed.flatten()[:PC_ref_dim]
    subprocess.run(['make', 'procrustes.o'], stdout=subprocess.PIPE)
    subprocess.run(['./procrustes.o'], stdout=subprocess.PIPE)
    R_trace = np.loadtxt('procrustes_A.dat')
    rho_trace = np.loadtxt('procrustes_rho.dat')
    c_trace = np.loadtxt('procrustes_c.dat')
    assert np.allclose(R_trace, R)
    assert np.allclose(rho_trace, rho)
    assert np.allclose(c_trace, c)
    # Test procrustes with different dimensions
    R_diffdim, rho_diffdim, c_diffdim = fp.procrustes_diffdim(PC_ref, PC_new_head)
    R_diffdim_trace = np.loadtxt('pprocrustes_A.dat')
    rho_diffdim_trace = np.loadtxt('pprocrustes_rho.dat')
    c_diffdim_trace = np.loadtxt('pprocrustes_c.dat')
    assert np.allclose(R_diffdim_trace, R_diffdim)
    assert np.allclose(rho_diffdim_trace, rho_diffdim)
    assert np.allclose(c_diffdim_trace, c_diffdim)
    print('Passed!')


def test_pca():
    pref_ref = '../data/kgn/kgn_bial_orphans_snps_ukb'
    # pref_ref = '../data/kgn/kgn_bial_orphans_snps_ukb_snps_comm'
    # pref_stu = '../data/ukb/ukb'
    # pref_stu = '../data/ukb/ukb_snps_comm'
    pref_stu = '../data/ukb/ukb_snps_kgn_2c'
    # pref_stu = '../data/ukb/ukb_snps_kgn_1k'
    popu_ref_filename = '../data/kgn/kgn_orphans.superpopu'
    pcs_trace_ref_filename = '../data/kgn_ukb_2c/kgn_ukb_2c.RefPC.coord'
    pcs_trace_stu_filename = '../data/kgn_ukb_2c/kgn_ukb_2c.ProPC.coord'
    dim_ref = 4
    log_level = 'info'
    use_memmap = False

    pcs_ref, pcs_stu_sp, popu_ref, popu_stu_pred_sp =  fp.run_pca(pref_ref, pref_stu, popu_ref_filename = popu_ref_filename, dim_ref=dim_ref, method='sp', use_memmap=use_memmap, log_level=log_level)
    pcs_ref, pcs_stu_ap, popu_ref, popu_stu_pred_ap =  fp.run_pca(pref_ref, pref_stu, popu_ref_filename = popu_ref_filename, dim_ref=dim_ref, method='ap', use_memmap=use_memmap, log_level=log_level)
    pcs_ref, pcs_stu_oadp, popu_ref, popu_stu_pred_oadp =  fp.run_pca(pref_ref, pref_stu, popu_ref_filename = popu_ref_filename, dim_ref=dim_ref, method='oadp', use_memmap=use_memmap, log_level=log_level)
    # pcs_ref, pcs_stu_oadp, popu_ref, popu_stu_pred_oadp =  fp.run_pca(pref_ref, pref_stu, popu_ref_filename = popu_ref_filename, dim_ref=dim_ref, method='adp', use_memmap=True, log_level=log_level)

    assert np.allclose(pcs_stu_ap, pcs_stu_oadp, 1e-1, 5e-2)

    pcs_ref_trace = fp.load_trace(pcs_trace_ref_filename, isref=True)
    pcs_stu_trace = fp.load_trace(pcs_trace_stu_filename)
    for i in range(dim_ref):
        corr = np.correlate(pcs_ref[:,i], pcs_ref_trace[:,i])
        sign = np.sign(corr)
        pcs_ref_trace[:,i] *= sign
        pcs_stu_trace[:,i] *= sign
    assert np.allclose(pcs_ref, pcs_ref_trace, 1e-3, 1e-3)
    assert np.allclose(pcs_stu_oadp, pcs_stu_trace, 1e-1, 5e-2)
    assert np.allclose(pcs_stu_ap, pcs_stu_trace, 1e-1, 5e-2)

    print('Procrustes similarity score (compared to TRACE result):')
    print('SP:')
    smlr_trace_sp = fp.procrustes_similarity(pcs_stu_trace, pcs_stu_sp)
    print(smlr_trace_sp)
    print('AP:')
    smlr_trace_ap = fp.procrustes_similarity(pcs_stu_trace, pcs_stu_ap)
    print(smlr_trace_ap)
    print('OADP:')
    smlr_trace_oadp = fp.procrustes_similarity(pcs_stu_trace, pcs_stu_oadp)
    print(smlr_trace_oadp)
    print('Ref:')
    smlr_trace_ref = fp.procrustes_similarity(pcs_ref_trace, pcs_ref)
    print(smlr_trace_ref)

    method_list = ['sp', 'adp', 'oadp']
    pcs_stu_list = [pcs_stu_sp, pcs_stu_ap, pcs_stu_oadp]
    popu_stu_list = [popu_stu_pred_oadp]*len(pcs_stu_list)
    fp.plot_pcs(pcs_ref, pcs_stu_list, popu_ref, popu_stu_list, method_list, out_pref=pref_stu)


test_online_svd_procrust()
test_pca()
