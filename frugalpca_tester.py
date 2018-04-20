import frugalpca as fp

ref_pref = '../data/kgn/kgn_snps_ukb'
stu_pref = '../data/ukb/ukb_snps_kgn_1k'
popu_ref_filename = '../data/kgn/kgn_popu.table'
superpopu_ref_filename = '../data/kgn/kgn_superpopu.table'
popu_col_name = 'Superpopulation'

log = fp.create_logger(stu_pref)
fp.test_online_svd_procrust()
fp.run_pca(ref_pref, stu_pref, popu_ref_filename, superpopu_ref_filename, popu_col_name)
