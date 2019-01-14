import pandas as pd
import numpy as np
import frugalpca as fp
import subprocess
import os.path

# def plot_results(pref_ref, pref_stu):
#     pcs_ref = np.loadtxt(pref_ref + '_ref.pcs')
#     pcs_stu_sp = np.loadtxt(pref_stu + '_stu_sp.pcs')
#     pcs_stu_ap = np.loadtxt(pref_stu + '_stu_ap.pcs')
#     pcs_stu_oadp = np.loadtxt(pref_stu + '_stu_oadp.pcs')
#     popu_ref = np.loadtxt(pref_ref + '.popu', dtype=np.object)[:,2]
#     popu_stu_pred_sp = np.loadtxt(pref_stu + '_pred_sp.popu', dtype=np.object)[:,2]
#     popu_stu_pred_ap = np.loadtxt(pref_stu + '_pred_ap.popu', dtype=np.object)[:,2]
#     popu_stu_pred_oadp = np.loadtxt(pref_stu + '_pred_oadp.popu', dtype=np.object)[:,2]
#     method_list = ['sp', 'ap', 'oadp']
#     pcs_stu_list = [pcs_stu_sp, pcs_stu_ap, pcs_stu_oadp]
#     popu_stu_list = [pcs_stu_pred_sp, pcs_stu_pred_ap, pcs_stu_pred_oadp]
#     fp.plot_pcs(pcs_ref, pcs_stu_list, popu_ref, popu_stu_list, method_list, out_pref=pref_stu)


def test_standardize():
    print('Testing standardize...')
    x = np.array([[0.0,1.0,2.0],
                  [2,2,2],
                  [2,3,2],
                  [0,1,3]])
    x_standardized_crct = np.array([[-1.22474487,0,1.22474487],
                                    [0,0,0],
                                    [0,0,0],
                                    [-1,1,0]])
    x_mean_crct = np.array([1,2,2,0.5]).reshape((-1,1))
    x_std_crct = np.array([0.81649658,1.,1.,0.5]).reshape((-1,1))
    x_mean, x_std = fp.standardize(x, miss=3)
    assert np.allclose(x, x_standardized_crct)
    assert np.allclose(x_mean, x_mean_crct)
    assert np.allclose(x_std, x_std_crct)
    print('Passed!')

    y = np.array([[ 1.,  0.,  1.],
                  [ 0., 3.,  0.],
                  [ 2.,  2.,  0.],
                  [3.,  0.,  0.]])
    y_standardized_crct = np.array([[ 0., -1.,  0.],
                                    [-1., 0, -1.],
                                    [ 0.,  0., -1.],
                                    [0, -1., -1.]])
    y_standardized_crct = np.array([[ 0.        , -1.22474487,  0.        ],
                                    [-2.        ,  0.        , -2.        ],
                                    [ 0.        ,  0.        , -2.        ],
                                    [ 0.        , -1.        , -1.        ]])
    fp.standardize(y, x_mean_crct, x_std_crct, miss=3)
    assert np.allclose(y, y_standardized_crct)

def plink_keep(bfile, keep, out):
    bashout = subprocess.run(
            ['plink', '--keep-allele-order', '--make-bed',
            '--indiv-sort', 'file', keep,
            '--bfile', bfile,
            '--keep', keep,
            '--out', out],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert bashout.stderr.decode('utf-8')[:5] is not 'Error'
    # keyword = bashout.stdout.decode('utf-8').split('\n')[-1][:5]
    # assert keyword is not 'Error'
    os.remove(out+'.log')

def plink_merge(bfile, bmerge, out):
    bashout = subprocess.run(
            ['plink', '--keep-allele-order', '--indiv-sort', 'none', '--make-bed',
            '--bfile', bfile,
            '--bmerge', bmerge,
            '--out', out],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert bashout.stderr.decode('utf-8')[:5] is not 'Error'
    os.remove(out+'.log')

def plink_remove(bfile, remove, out):
    bashout = subprocess.run(
        ['plink', '--keep-allele-order', '--indiv-sort', 'none', '--make-bed',
        '--bfile', bfile,
        '--remove', remove,
        '--out', out],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert bashout.stderr.decode('utf-8')[:5] is not 'Error'
    os.remove(out+'.log')

def get_homogen(ref_merged_filepref, homogen_dist_threshold=3):
    merged_popu_df = pd.read_table(ref_merged_filepref+'.popu', header=None)
    merged_popu_df.columns = ['fid', 'iid', 'popu']
    merged_coord = fp.run_pca(ref_merged_filepref, None, method='sp', load_saved_ref_decomp=False, plot_results=True)[0]
    dim_ref = merged_coord.shape[1]
    merged_coord_df = pd.DataFrame(merged_coord)
    merged_coord_df = pd.DataFrame(merged_coord)
    merged_df = pd.concat([merged_popu_df, merged_coord_df], axis=1)
    merged_popu = merged_df['popu']
    merged_popu_uniq = np.unique(merged_popu)
    merged_inlier_df = merged_df[0:0]
    for pp in merged_popu_uniq:
        if pp[-8:] == 'borrowed':
            pp_base = pp[:3]
            pp_ref_df = merged_df[merged_popu == pp_base]
            pp_stu_df = merged_df[merged_popu == pp]
            pp_ref_coord = merged_coord[merged_popu == pp_base]
            pp_stu_coord = merged_coord[merged_popu == pp]
            n_ref = pp_ref_coord.shape[0]
            n_stu = pp_stu_coord.shape[0]
            mn, std, U, s, V, pp_ref_pcs = fp.pca_ref(pp_ref_coord.T, dim_ref=dim_ref)
            se = np.std(pp_ref_pcs, axis=0)
            pp_stu_pcs = fp.pca_stu(pp_stu_coord.T, mn, std, 'sp', U)
            pp_stu_dist = np.sqrt(np.sum(pp_stu_pcs**2 / se**2, axis=1))
            pp_stu_isin = pp_stu_dist < homogen_dist_threshold
            pp_stu_inlier_df = pp_stu_df[pp_stu_isin]
            # if not remember_borrowed:
            #    pp_stu_inlier_df['popu'] = pp_base
            merged_inlier_df = pd.concat((merged_inlier_df, pp_stu_inlier_df), axis=0)
            # print(se[:4])
            # pp_stu_popu = pp_stu_isin
            # fp.plot_pcs(pp_ref_pcs, pp_stu_pcs, popu_stu=pp_stu_isin, method='sp', out_pref=ref_merged_filepref+'_'+pp_base)
        else:
            pp_ref_df = merged_df[merged_popu == pp]
            merged_inlier_df = pd.concat((merged_inlier_df, pp_ref_df), axis=0)
    # print(merged_df.groupby('popu').count().iloc[:,0])
    # print('Number of samples: ' + str(merged_df.shape[0]))
    # print(merged_inlier_df.groupby('popu').count().iloc[:,0])
    # print('Number of inliers: ' + str(merged_inlier_df.shape[0]))
    print('Samples left: ' + str(merged_inlier_df.shape[0]) + '/' + str(merged_df.shape[0]))
    merged_inlier_df.iloc[:,:3].to_csv(ref_merged_filepref+'.popu', sep='\t', header=False, index=False)
    plink_keep(ref_merged_filepref, ref_merged_filepref+'.popu', ref_merged_filepref)
    # if os.path.isfile(ref_merged_filepref+'_ref.pcs'):
    #     os.remove(ref_merged_filepref+'_ref.pcs')
    if merged_inlier_df.shape[0] != merged_df.shape[0]:
        get_homogen(ref_merged_filepref)

def test_online_svd_procrust():
    np.random.seed(21)
    print('Testing svd_online and procrustes...')

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
    np.savetxt('tmp/test_X.dat', X)
    np.savetxt('tmp/test_b.dat', b)

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
    np.savetxt('tmp/test_PC_ref.dat', PC_ref)
    np.savetxt('tmp/test_PC_ref_fat.dat', PC_ref_fat)
    np.savetxt('tmp/test_PC_new_head.dat', PC_new_head)
    # Test procrustes with the same dimension
    R, rho, c = fp.procrustes(PC_ref_fat, PC_new_head)
    # PC_new_tail_trsfed = PC_new_tail @ R * rho + c
    # PC_new_tail_trsfed = PC_new_tail_trsfed.flatten()[:PC_ref_dim]
    subprocess.run(['make', 'procrustes.o'], stdout=subprocess.PIPE)
    subprocess.run(['./procrustes.o'], stdout=subprocess.PIPE)
    R_trace = np.loadtxt('tmp/procrustes_A.dat')
    rho_trace = np.loadtxt('tmp/procrustes_rho.dat')
    c_trace = np.loadtxt('tmp/procrustes_c.dat')
    assert np.allclose(R_trace, R)
    assert np.allclose(rho_trace, rho)
    assert np.allclose(c_trace, c)
    # Test procrustes with different dimensions
    R_diffdim, rho_diffdim, c_diffdim = fp.procrustes_diffdim(PC_ref, PC_new_head)
    R_diffdim_trace = np.loadtxt('tmp/pprocrustes_A.dat')
    rho_diffdim_trace = np.loadtxt('tmp/pprocrustes_rho.dat')
    c_diffdim_trace = np.loadtxt('tmp/pprocrustes_c.dat')
    assert np.allclose(R_diffdim_trace, R_diffdim)
    assert np.allclose(rho_diffdim_trace, rho_diffdim)
    assert np.allclose(c_diffdim_trace, c_diffdim)
    print('Passed!')


def test_pca_allmeths(pref_ref, pref_stu, cmp_trace=True, load_pcs=False, load_popu=False, assert_results=False, pref_out_trace=None, pref_ref_trace=None, dim_ref=4, plot_dim=4, plot_size=None, plot_title=None, plot_color_stu=None, plot_legend=True, alpha_stu=0.5, alpha_ref=0.1, plot_centers=False, hdpca_n_spikes=None, n_clusters=None, load_saved_ref_decomp=False):
    print('Reference: ' + pref_ref)
    print('Study: ' + pref_stu)
    popu_filename_ref = pref_ref + '.popu'
    subpopu_filename_ref = pref_ref + '_sub.popu'
    base_ref = os.path.basename(pref_ref)
    dir_stu = os.path.dirname(pref_stu)
    pref_out = pref_stu + '_sturef_' + base_ref
    if pref_out_trace is None:
        pref_out_trace = pref_out
    pcs_trace_ref_filename = pref_out_trace + '.RefPC.coord'
    pcs_trace_stu_filename = pref_out_trace + '.ProPC.coord'
    log_level = 'info'
    use_memmap = False

    if load_pcs:
        pcs_ref = np.loadtxt(pref_ref + '_ref.pcs')
        pcs_stu_ap = np.loadtxt(pref_out + '_stu_ap.pcs')
        pcs_stu_sp = np.loadtxt(pref_out + '_stu_sp.pcs')
        pcs_stu_oadp = np.loadtxt(pref_out + '_stu_oadp.pcs')

        if n_clusters is None:
            popu_ref = np.loadtxt(pref_ref + '.popu', dtype=np.object)[:,2]
        else:
            fam_ref = pd.read_table(pref_ref+'.fam', header=None, delimiter=' ').iloc[:,:2]
            assert fam_ref.shape[0] == pcs_ref.shape[0]
            popu_ref = fp.pred_popu_ref(pcs_ref, n_clusters, out_filename=pref_ref+'_pred.popu', rowhead_df=fam_ref)

        if load_popu:
            popu_stu_pred_ap = np.loadtxt(pref_out + '_pred_ap.popu', dtype=np.object)[:,2]
            popu_stu_pred_sp = np.loadtxt(pref_out + '_pred_sp.popu', dtype=np.object)[:,2]
            popu_stu_pred_oadp = np.loadtxt(pref_out + '_pred_oadp.popu', dtype=np.object)[:,2]
        else:
            fam_stu = pd.read_table(pref_stu+'.fam', header=None, delimiter=' ').iloc[:,:2]
            assert fam_stu.shape[0] == pcs_stu_sp.shape[0]
            popu_stu_pred_ap = fp.pred_popu_stu(pcs_ref, popu_ref, pcs_stu_ap, pref_out+'_pred_ap.popu', fam_stu)[0]
            popu_stu_pred_sp = fp.pred_popu_stu(pcs_ref, popu_ref, pcs_stu_sp, pref_out+'_pred_sp.popu', fam_stu)[0]
            popu_stu_pred_oadp = fp.pred_popu_stu(pcs_ref, popu_ref, pcs_stu_oadp, pref_out+'_pred_oadp.popu', fam_stu)[0]
    else:
        pcs_ref, pcs_stu_ap, popu_ref, popu_stu_pred_ap = fp.run_pca(pref_ref, pref_stu, method='ap', dim_ref=dim_ref, hdpca_n_spikes=hdpca_n_spikes, load_saved_ref_decomp=load_saved_ref_decomp)[:4]
        pcs_ref, pcs_stu_sp, popu_ref, popu_stu_pred_sp = fp.run_pca(pref_ref, pref_stu, method='sp', dim_ref=dim_ref, load_saved_ref_decomp=load_saved_ref_decomp)[:4]
        pcs_ref, pcs_stu_oadp, popu_ref, popu_stu_pred_oadp = fp.run_pca(pref_ref, pref_stu, method='oadp', dim_ref=dim_ref, load_saved_ref_decomp=load_saved_ref_decomp)[:4]


    pcs_min = np.vstack((pcs_ref, pcs_stu_ap)).min(axis=0)
    pcs_max = np.vstack((pcs_ref, pcs_stu_ap)).max(axis=0)
    plot_lim = np.vstack((pcs_min, pcs_max)) * 1.20
    fp.plot_pcs(pcs_ref, pcs_stu_ap, popu_ref, popu_stu_pred_ap, 'ap', out_pref=pref_out, plot_lim=plot_lim, plot_dim=plot_dim, plot_size=plot_size, plot_title=plot_title, plot_color_stu=plot_color_stu, plot_legend=plot_legend, alpha_stu=alpha_stu, alpha_ref=alpha_ref, plot_centers=plot_centers)
    fp.plot_pcs(pcs_ref, pcs_stu_sp, popu_ref, popu_stu_pred_sp, 'sp', out_pref=pref_out, plot_lim=plot_lim, plot_dim=plot_dim, plot_size=plot_size, plot_title=plot_title, plot_color_stu=plot_color_stu, plot_legend=plot_legend, alpha_stu=alpha_stu, alpha_ref=alpha_ref, plot_centers=plot_centers)
    fp.plot_pcs(pcs_ref, pcs_stu_oadp, popu_ref, popu_stu_pred_oadp, 'oadp', out_pref=pref_out, plot_lim=plot_lim, plot_dim=plot_dim, plot_size=plot_size, plot_title=plot_title, plot_color_stu=plot_color_stu, plot_legend=plot_legend, alpha_stu=alpha_stu, alpha_ref=alpha_ref, plot_centers=plot_centers)

    if cmp_trace:
        pcs_ref_trace = fp.load_trace(pcs_trace_ref_filename, isref=True)
        pcs_stu_trace = fp.load_trace(pcs_trace_stu_filename)
        for i in range(dim_ref):
            corr = np.correlate(pcs_ref[:,i], pcs_ref_trace[:,i])
            sign = np.sign(corr)
            pcs_ref_trace[:,i] *= sign
            pcs_stu_trace[:,i] *= sign
        popu_stu_pred_trace = fp.pred_popu_stu(pcs_ref_trace, popu_ref, pcs_stu_trace)[0]
        fp.plot_pcs(pcs_ref, pcs_stu_trace, popu_ref, popu_stu_pred_trace, 'adp', out_pref=pref_out, plot_lim=plot_lim, plot_dim=plot_dim, plot_size=plot_size, plot_title=plot_title, plot_color_stu=plot_color_stu, plot_legend=plot_legend, alpha_stu=alpha_stu, alpha_ref=alpha_ref, plot_centers=plot_centers)

    if cmp_trace:
        print('Accuracy comparison (procrustes, ADP, geocenter): ')
        print('Ref:')
        print(fp.procrustes_similarity(pcs_ref_trace, pcs_ref))
        print(np.linalg.norm(pcs_ref_trace - pcs_ref))
        print('SP:')
        print(fp.procrustes_similarity(pcs_stu_trace, pcs_stu_sp))
        print(np.linalg.norm(pcs_stu_trace - pcs_stu_sp))
        print(fp.geocenter_similarity(pcs_stu_sp, popu_stu_pred_sp, pcs_ref, popu_ref))
        print('AP:')
        print(fp.procrustes_similarity(pcs_stu_trace, pcs_stu_ap))
        print(np.linalg.norm(pcs_stu_trace - pcs_stu_ap))
        print(fp.geocenter_similarity(pcs_stu_ap, popu_stu_pred_ap, pcs_ref, popu_ref))
        print('OADP:')
        print(fp.procrustes_similarity(pcs_stu_trace, pcs_stu_oadp))
        print(np.linalg.norm(pcs_stu_trace - pcs_stu_oadp))
        print(fp.geocenter_similarity(pcs_stu_oadp, popu_stu_pred_oadp, pcs_ref, popu_ref))
        print('ADP:')
        print(fp.procrustes_similarity(pcs_stu_trace, pcs_stu_trace))
        print(np.linalg.norm(pcs_stu_trace - pcs_stu_trace))
        print(fp.geocenter_similarity(pcs_stu_trace, popu_stu_pred_trace, pcs_ref, popu_ref))

        if assert_results:
            assert smlr_trace_ref > 0.99
            assert smlr_trace_sp > 0.99
            assert smlr_trace_ap > 0.99
            assert smlr_trace_oadp > 0.99
            assert np.allclose(pcs_ref, pcs_ref_trace, 1e-3, 1e-3)
            assert np.allclose(pcs_stu_sp, pcs_stu_trace, 5e-1, 5e-2)
            assert np.allclose(pcs_stu_ap, pcs_stu_trace, 1e-1, 5e-2)
            assert np.allclose(pcs_stu_oadp, pcs_stu_trace, 1e-1, 5e-2)

def test_pca_subpopu(cmp_trace=False, load_pcs=False, plot_size=None, plot_title=None):
    pref_ref = '../data/kgn/kgn_bial_orphans_snps_ukb_snpscap_ukb'
    pref_stu = '../data/ukb/ukb_snpscap_kgn_bial_orphans'
    popu_name_this = 'AFR'
    bashout=subprocess.run(['bash', 'keep_popu.sh', pref_ref, pref_stu, popu_name_this], stdout=subprocess.PIPE)
    pref_ref_this, pref_stu_this = bashout.stdout.decode('utf-8').split('\n')[-3:-1]
    test_pca_allmeths(pref_ref_this, pref_stu_this, cmp_trace=cmp_trace, load_pcs=load_pcs, plot_size=plot_size, plot_title=plot_title, hdpca_n_spikes=4)

def convert_ggsim(i):
    n = 1000 + i * 500
    filepref = '../data/ggsim'+str(n)+'/ggsim'+str(n)+'_100000_'+str(n)+'_2_1_100_0'
    fp.trace2bed(filepref)
    filepref = '../data/ggsim'+str(n)+'/ggsim'+str(n)+'_100000_'+'200'+'_2_1_100_1'
    fp.trace2bed(filepref)

def test_pca_500k_EUR():
    pref_ref = '../data/kgn/kgn_bial_orphans_snps_ukb_snpscap_ukb'
    pref_stu = '../data/ukb/ukb_snpscap_kgn_bial_orphans'
    test_pca_subpopu(pref_ref, pref_stu, 'EUR', cmp_trace=False)

def test_pca_array(pref_ref, pref_stu, method, n_chunks, i):
    if type(method) is int:
        method = {0:'oadp', 1:'ap', 2:'sp'}[method]
    pref_stu_this = fp.split_bed_indiv(pref_stu, n_chunks, i)
    fp.run_pca(pref_ref, pref_stu_this, method=method)

def test_merge_array_results():

    # n_chunks = 10
    # ref_filepref = '../data/kgn/kgn_bial_orphans_snps_ukb_snpscap_ukb'
    # stu_filepref = '../data/ukb/ukb_snpscap_kgn_bial_orphans_5c_nchunks10/ukb_snpscap_kgn_bial_orphans_5c_nchunks10'
    # cmp_trace=True
    # traceout_filepref = '../data/ukb/ukb_snpscap_kgn_bial_orphans_5c_sturef_kgn_bial_orphans_snps_ukb_snpscap_ukb'

    # n_chunks = 100
    # ref_filepref = '../data/kgn/kgn_bial_orphans_snps_ukb_snpscap_ukb'
    # stu_filepref = '../data/ukb/ukb_snpscap_kgn_bial_orphans_nchunks100/ukb_snpscap_kgn_bial_orphans_nchunks100'
    # cmp_trace=False
    # traceout_filepref=None

    n_chunks = 100
    ref_filepref = '../data/kgn/kgn_bial_orphans_snps_ukb_snpscap_ukb_EUR'
    stu_filepref = '../data/ukb/ukb_snpscap_kgn_bial_orphans_pred_EUR_nchunks100/ukb_snpscap_kgn_bial_orphans_pred_EUR_nchunks100'
    cmp_trace=False
    traceout_filepref=None
    # test_pca_allmeths(ref_filepref, stu_filepref, cmp_trace=cmp_trace, load_pcs=True, load_popu=False, assert_results=False, plot_centers=False, plot_size=(12,4), hdpca_n_spikes=4)

    n_chunks = 100
    ref_filepref = '../data/kgn/kgn_bial_orphans_snps_ukb_snpscap_ukb_EUR_merge_ukb_EUR_pure_homogen'
    stu_filepref = '../data/ukb/ukb_snpscap_kgn_bial_orphans_pred_EUR_nchunks100/ukb_snpscap_kgn_bial_orphans_pred_EUR_heterogen_nchunks100/ukb_snpscap_kgn_bial_orphans_pred_EUR_heterogen_nchunks100'
    cmp_trace=False
    traceout_filepref=None
 
    # fp.merge_array_results(ref_filepref, stu_filepref, 'sp', n_chunks)
    # fp.merge_array_results(ref_filepref, stu_filepref, 'ap', n_chunks)
    # fp.merge_array_results(ref_filepref, stu_filepref, 'oadp', n_chunks)
    test_pca_allmeths(ref_filepref, stu_filepref, cmp_trace=cmp_trace, load_pcs=True, load_popu=False, assert_results=False, plot_centers=False, plot_size=(12,4), hdpca_n_spikes=4)

def test_split_bed_indiv():
    filepref = '../data/ukb/ukb_snpscap_kgn_bial_orphans_5c'
    n_chunks = 10
    i = 9
    fp.split_bed_indiv(filepref, n_chunks, i)

def test_pca_ggsim(i):
    i = int(i)
    n = 1000 + i * 500
    pref_ref = '../data/ggsim'+str(n)+'/ggsim'+str(n)+'_100000_'+str(n)+'_2_1_100_0'
    pref_stu = '../data/ggsim'+str(n)+'/ggsim'+str(n)+'_100000_'+'200'+'_2_1_100_1'
    pref_out_trace = '../data/ggsim'+str(n)+'/ggsim'+str(n)+'_100000_'+str(n)+'_200'+'_2_1_100'
    test_pca_allmeths(pref_ref, pref_stu, pref_out_trace=pref_out_trace,
                        cmp_trace=True, load_saved_ref_decomp=False, load_pcs=False,
                        dim_ref=2, 
                        # dpca_n_spikes=4,
                        plot_title=str(n), plot_size=(6,6), plot_color_stu='k', plot_legend=True, alpha_stu=0.9, plot_dim=2, plot_centers=True)

def test_pca_ukb_5c():
    ref_filepref = '../data/kgn/kgn_bial_orphans_snps_ukb_snpscap_ukb'
    stu_filepref = '../data/ukb/ukb_snpscap_kgn_bial_orphans_5c'
    fp.run_pca(ref_filepref, None, method='sp', load_saved_ref_decomp=True, plot_results=True, alpha_ref=0.4, plot_size=(12,4))
    test_pca_allmeths(ref_filepref, stu_filepref, cmp_trace=True, load_pcs=True, load_popu=True, plot_size=(12,4), hdpca_n_spikes=4, alpha_ref=0.01, alpha_stu=0.8)

def test_pca_ukb_EUR_5c():
    ref_filepref = '../data/kgn/kgn_bial_orphans_snps_ukb_snpscap_ukb_EUR'
    stu_filepref = '../data/ukb/ukb_snpscap_kgn_bial_orphans_nchunks100/ukb_snpscap_kgn_bial_orphans_pred_EUR_oadp_5c'
    fp.run_pca(ref_filepref, None, method='sp', load_saved_ref_decomp=True, plot_results=True, alpha_ref=0.4, plot_size=(12,4))
    test_pca_allmeths(ref_filepref, stu_filepref, cmp_trace=True, load_pcs=True, load_popu=True, assert_results=False, plot_size=(12,4), hdpca_n_spikes=4, load_saved_ref_decomp=False, alpha_ref=0.01, alpha_stu=0.8)

def test_pca_EUR_pure():
    ref_filepref = '../data/kgn/kgn_bial_orphans_snps_ukb_snpscap_ukb_EUR'
    stu_filepref = '../data/ukb/ukb_snpscap_kgn_bial_orphans_pred_EUR_pure'
    test_pca_allmeths(ref_filepref, stu_filepref, cmp_trace=False, load_pcs=False, load_popu=False, assert_results=False, plot_size=(12,4), hdpca_n_spikes=4)

def test_pca_5c_EUR_impure_predrefpopu():
    ref_filepref = '../data/kgn/kgn_bial_orphans_snps_ukb_snpscap_ukb_EUR_merge_ukb_EUR_pure'
    stu_filepref = '../data/ukb/ukb_snpscap_kgn_bial_orphans_5c_pred_EUR_impure'
    test_pca_allmeths(ref_filepref, stu_filepref, cmp_trace=False, load_pcs=False, load_popu=False, assert_results=False, plot_size=(12,4), hdpca_n_spikes=4, n_clusters=5)

def test_pca_EUR_pure_homogen():
    ref_filepref = '../data/kgn/kgn_bial_orphans_snps_ukb_snpscap_ukb_EUR'
    stu_filepref = '../data/ukb/ukb_snpscap_kgn_bial_orphans_pred_EUR_pure_homogen'
    test_pca_allmeths(ref_filepref, stu_filepref, cmp_trace=False, load_pcs=False, load_popu=False, assert_results=False, plot_size=(12,4), hdpca_n_spikes=4)


def test_pca_5c_EUR_heterogen_ref_homogen():
    ref_filepref = '../data/kgn/kgn_bial_orphans_snps_ukb_snpscap_ukb_EUR_merge_ukb_EUR_pure_homogen'
    stu_filepref = '../data/ukb/ukb_snpscap_kgn_bial_orphans_5c_pred_EUR_heterogen'
    test_pca_allmeths(ref_filepref, stu_filepref, cmp_trace=False, load_pcs=False, load_popu=False, assert_results=False, plot_size=(12,4), hdpca_n_spikes=4)

def test_pca_5c_EUR_impure():
    ref_filepref = '../data/kgn/kgn_bial_orphans_snps_ukb_snpscap_ukb_EUR_withloan'
    stu_filepref = '../data/ukb/ukb_snpscap_kgn_bial_orphans_5c_pred_EUR_impure'
    test_pca_allmeths(ref_filepref, stu_filepref, cmp_trace=False, load_pcs=False, load_popu=False, assert_results=False, plot_size=(12,4), hdpca_n_spikes=4, load_saved_ref_decomp=False)

def test_add_pure_stu(popu):
    stu_filepref = '../data/ukb/ukb_snpscap_kgn_bial_orphans_pred_' + popu
    ref_filepref = '../data/kgn/kgn_bial_orphans_snps_ukb_' + popu + '_deol'
    merged_filepref = fp.add_pure_stu(ref_filepref, stu_filepref, n_pure_samples=1000, popu_purity_threshold=0.75, homogen_dist_threshold=2, n_iter_max=10)
    print('Removing selected samples from the study set...')
    # stu_filepref = '../data/ukb/ukb_snpscap_kgn_bial_orphans_5c_pred_EUR'
    out = stu_filepref + '_impure'
    plink_remove(stu_filepref, merged_filepref+'.popu', stu_filepref+'_impure')

    # # Remove pure study samples from study samples
    # n_chunks = 100
    # stu_filepref_chunk = '../data/ukb/ukb_snpscap_kgn_bial_orphans_pred_EUR_nchunks100/ukb_snpscap_kgn_bial_orphans_pred_EUR_nchunks100'
    # impure_filepref_chunk = '../data/ukb/ukb_snpscap_kgn_bial_orphans_pred_EUR_nchunks100/ukb_snpscap_kgn_bial_orphans_pred_EUR_impure_nchunks100'
    # for i in range(n_chunks):
    #     stu_filepref_chunk_this = stu_filepref_chunk + '_' + str(i).zfill(fp.SAMPLE_SPLIT_PREF_LEN)
    #     os.makedirs(heterogen_filepref_chunk, exist_ok=True)
    #     heterogen_filepref_chunk_this = heterogen_filepref_chunk + '/' + os.path.basename(heterogen_filepref_chunk) + '_' + str(i).zfill(fp.SAMPLE_SPLIT_PREF_LEN)
    #     plink_remove(stu_filepref_chunk_this, pure_filepref+'.popu', heterogen_filepref_chunk_this)


def test_pca_onepopu(popu, deol=False):
    if popu == "MIX":
        ref_filepref = '../data/kgn/kgn_bial_orphans_snps_ukb_snpscap_ukb'
    else:
        ref_filepref = '../data/kgn/kgn_bial_orphans_snps_ukb_' + popu
        if deol:
            ref_filepref += '_deol'
    stu_filepref = '../data/ukb/ukb_snpscap_kgn_bial_orphans_pred_' + popu + '_oadp'
    fp.run_pca(ref_filepref, None, method='sp', load_saved_ref_decomp=False, plot_results=True, alpha_ref=0.4, plot_size=(12,4))
    test_pca_allmeths(ref_filepref, stu_filepref, cmp_trace=False, load_pcs=True, load_popu=True, assert_results=False, plot_size=(12,4), hdpca_n_spikes=4, load_saved_ref_decomp=False, alpha_ref=0.01, alpha_stu=0.5)

def test_pca_ref(popu):
    # ref_filepref = '../data/kgn/kgn_bial_orphans_snps_ukb_' + popu
    # fp.run_pca(ref_filepref, None, method='sp', load_saved_ref_decomp=False, plot_results=True, alpha_ref=0.4, plot_size=(12,4))
    # ref_filepref = '../data/kgn/kgn_bial_orphans_snps_ukb_' + popu + '_deol'
    # fp.run_pca(ref_filepref, None, method='sp', load_saved_ref_decomp=False, plot_results=True, alpha_ref=0.4, plot_size=(12,4))
    # ref_filepref = '../data/kgn/kgn_bial_orphans_snps_ukb_' + popu + '_deol_withloan'
    # fp.run_pca(ref_filepref, None, method='sp', load_saved_ref_decomp=False, plot_results=True, alpha_ref=0.4, plot_size=(12,4))

    ref_filepref = '../data/kgn/kgn_bial_orphans_snps_ukb_ASW'
    fp.run_pca(ref_filepref, None, method='sp', load_saved_ref_decomp=False, plot_results=True, alpha_ref=0.4, plot_size=(12,4))
    ref_filepref = '../data/kgn/kgn_bial_orphans_snps_ukb_noASW'
    fp.run_pca(ref_filepref, None, method='sp', load_saved_ref_decomp=False, plot_results=True, alpha_ref=0.4, plot_size=(12,4))
    ref_filepref = '../data/kgn/kgn_bial_orphans_snps_ukb_AFR_norels'
    fp.run_pca(ref_filepref, None, method='sp', load_saved_ref_decomp=False, plot_results=True, alpha_ref=0.4, plot_size=(12,4))

def test_add_pure_stu_onepopu(popu):
    stu_filepref = '../data/ukb/ukb_snpscap_kgn_bial_orphans_pred_' + popu
    ref_filepref = '../data/kgn/kgn_bial_orphans_snps_ukb_' + popu + '_deol'
    merged_filepref = fp.add_pure_stu(ref_filepref, stu_filepref, n_pure_samples=1, popu_purity_threshold=0.75, homogen_dist_threshold=3, n_iter_max=10)
    print('Removing selected samples from the study set...')
    out = stu_filepref + '_impure'
    plink_remove(stu_filepref, merged_filepref+'.popu', stu_filepref+'_impure')

def test_pca_ref_withloan(popu):
    ref_filepref = '../data/kgn/kgn_bial_orphans_snps_ukb_' + popu + '_deol_withloan'
    fp.run_pca(ref_filepref, None, method='sp', load_saved_ref_decomp=False, plot_results=True, plot_size=(12,4))

def test_run_pca(popu=None):
    # ref_filepref = '../data/kgn/kgn_bial_orphans_snps_ukb_' + popu
    # stu_filepref = '../data/ukb/ukb_snpscap_kgn_bial_orphans_pred_' + popu
    # fp.run_pca(ref_filepref, stu_filepref, method='ap', plot_size=(12,4), hdpca_n_spikes=4, load_saved_ref_decomp=False, alpha_ref=0.1, alpha_stu=0.5, plot_results=True)
    # ref_filepref = '../data/kgn/kgn_bial_orphans_snps_ukb_' + popu + '_deol'
    # stu_filepref = '../data/ukb/ukb_snpscap_kgn_bial_orphans_pred_' + popu
    # fp.run_pca(ref_filepref, stu_filepref, method='ap', plot_size=(12,4), hdpca_n_spikes=4, load_saved_ref_decomp=False, alpha_ref=0.1, alpha_stu=0.5, plot_results=True)
    # ref_filepref = '../data/kgn/kgn_bial_orphans_snps_ukb_' + popu 
    # stu_filepref = '../data/ukb/ukb_snpscap_kgn_bial_orphans_pred_' + popu
    ref_filepref = '../data/kgn/kgn_bial_snpscap_ukb'
    stu_filepref = '../data/ukb/ukb_snpscap_kgn_bial_orphans_5c'
    fp.run_pca(ref_filepref, stu_filepref, method='ap', plot_size=(12,4), hdpca_n_spikes=4, load_saved_ref_decomp=True, alpha_ref=0.1, alpha_stu=0.5, plot_results=True)

def test_plot_results():
    # ref_filepref = '../data/kgn/kgn_bial_orphans_snps_ukb_snpscap_ukb'
    # ref_filepref = '../data/kgn/kgn_bial_orphans_snps_ukb_snpscap_ukb_EUR'
    # stu_filepref = '../data/ukb/ukb_snpscap_kgn_bial_orphans_nchunks100/ukb_snpscap_kgn_bial_orphans_nchunks100'
    # stu_filepref = '../data/ukb/ukb_snpscap_kgn_bial_orphans_nchunks100/ukb_snpscap_kgn_bial_orphans_sturef_kgn_bial_orphans_snps_ukb_snpscap_ukb'
    # stu_filepref = '../data/ukb/ukb_snpscap_kgn_bial_orphans_nchunks100/EUR_nchunks100/ukb_snpscap_kgn_bial_orphans_pred_EUR_oadp_sturef_kgn_bial_orphans_snps_ukb_snpscap_ukb_EUR'
    # stu_filepref = '../data/ukb/ukb_snpscap_kgn_bial_orphans_pred_EUR_nchunks100/ukb_snpscap_kgn_bial_orphans_pred_EUR_nchunks100_sturef_kgn_bial_orphans_snps_ukb_snpscap_ukb_EUR'
    # pref_out = '../data/ukb/ukb_snpscap_kgn_bial_orphans_nchunks100/ukb_snpscap_kgn_bial_orphans_nchunks100'
    pref_out = stu_filepref
    plot_lim = np.array([[-60, -80, -125, -200], [90, 110, 125, 300]], dtype=np.float32) 

    pcs_ref = np.loadtxt(ref_filepref + '_ref.pcs')
    popu_ref = np.loadtxt(ref_filepref + '.popu', dtype=object, usecols=2)
    fp.plot_pcs(pcs_ref, None, popu_ref, None, 'ref', out_pref=pref_out, plot_size=(12,4), alpha_ref=0.4, plot_lim=plot_lim)
    for method in ['sp', 'ap', 'oadp']:
        pcs_stu = np.loadtxt(stu_filepref + '_stu_' + method + '.pcs')
        popu_stu = np.loadtxt(stu_filepref + '_pred_' + method + '.popu', dtype=object, usecols=2)
        fp.plot_pcs(pcs_ref, pcs_stu, popu_ref, popu_stu, method, out_pref=pref_out, plot_size=(12,4), alpha_ref=0.1, alpha_stu=0.4, plot_lim=plot_lim)

def test_pred_popu():
    ref_filepref = '../data/kgn/kgn_bial_snpscap_ukb'
    stu_filepref = '../data/ukb/ukb_snpscap_kgn_bial_orphans_nchunks100/ukb_snpscap_kgn_bial_orphans_nchunks100'
    out_filename = '../data/ukb/ukb_snpscap_kgn_bial_orphans_nchunks100/bydist.popu'
    pcs_ref = np.loadtxt(ref_filepref + '_ref.pcs')
    popu_ref = np.loadtxt(ref_filepref + '.popu', usecols=2, dtype=object)
    pcs_stu = np.loadtxt(stu_filepref + '_stu_ap.pcs')
    popu_stu, popu_stu_proba, popu_stu_dist = fp.pred_popu_stu(pcs_ref, popu_ref, pcs_stu, out_filename=out_filename, weights='distance')
    
def test_pca_diffpopu(popu_ref, popu_stu, deol=False):
    ref_filepref = '../data/kgn/kgn_bial_orphans_snps_ukb_' + popu_ref
    if deol:
        ref_filepref += '_deol'
    if popu_stu == 'EUR':
        stu_filepref = '../data/ukb/ukb_snpscap_kgn_bial_orphans_nchunks100/ukb_snpscap_kgn_bial_orphans_pred_EUR_oadp_5c'
    else:
        stu_filepref = '../data/ukb/ukb_snpscap_kgn_bial_orphans_pred_' + popu_stu + '_oadp'
    pref_out_trace = '../data/ukb/ukb_' + popu_stu + '_sturef_kgn_' + popu_ref 
    test_pca_allmeths(ref_filepref, stu_filepref, cmp_trace=False, load_pcs=False, load_popu=False, assert_results=False, load_saved_ref_decomp=True, plot_size=(12,4), hdpca_n_spikes=4, alpha_ref=0.1, alpha_stu=0.5)

def test_pca_children():
    ref_filepref = '../data/kgn/kgn_bial_orphans_snps_ukb_snpscap_ukb'
    stu_filepref = '../data/kgn/kgn_bial_snpscap_ukb_children'
    pref_out_trace = '../data/kgn/kgn_children_sturef_childless'
    fp.run_pca(ref_filepref, None, method='ref', load_saved_ref_decomp=False, plot_results=True, alpha_ref=0.4, plot_size=(12,4))
    test_pca_allmeths(ref_filepref, stu_filepref, cmp_trace=True, load_pcs=True, load_popu=False, assert_results=False, load_saved_ref_decomp=True, plot_size=(12,4), hdpca_n_spikes=4, alpha_ref=0.001, alpha_stu=0.9, pref_out_trace=pref_out_trace)
