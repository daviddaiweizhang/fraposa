import pandas as pd
import frugalpca as fp

pref_ref = '../data/kgn/kgn_snps_ukb'
pref_stu = '../data/ukb/ukb_snps_kgn_1k'
popu_ref_filename = '../data/kgn/kgn_orphans.superpopu'
# log = fp.create_logger(pref_stu, 'debug')
dim_ref = 4

fp.test_online_svd_procrust()
pcs_ref, pcs_stu_ap, popu_ref, popu_stu_pred_ap =  fp.run_pca(pref_ref, pref_stu, popu_ref_filename = popu_ref_filename, dim_ref=dim_ref, method='ap', use_memmap=False)
pcs_ref, pcs_stu_sp, popu_ref, popu_stu_pred_sp =  fp.run_pca(pref_ref, pref_stu, popu_ref_filename = popu_ref_filename, dim_ref=dim_ref, method='sp', use_memmap=False)
pcs_ref, pcs_stu_oadp, popu_ref, popu_stu_pred_oadp =  fp.run_pca(pref_ref, pref_stu, popu_ref_filename = popu_ref_filename, dim_ref=dim_ref, method='oadp', use_memmap=False)
pcs_stu_list = [pcs_stu_sp, pcs_stu_ap, pcs_stu_oadp]
popu_stu_list = [popu_stu_pred_oadp]*3
method_list = ['sp', 'adp', 'oadp']
fp.plot_pcs(pcs_ref, pcs_stu_list, popu_ref, popu_stu_list, method_list, out_pref=pref_stu)
