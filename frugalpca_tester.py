import pandas as pd
import frugalpca as fp

pref_ref = '../data/kgn/kgn_snps_ukb'
pref_stu = '../data/ukb/ukb_snps_kgn_1k'
popu_ref_filename = '../data/kgn/kgn_orphans.superpopu'
# log = fp.create_logger(pref_stu, 'debug')

fp.test_online_svd_procrust()

# fp.run_pca(pref_ref, pref_stu, popu_ref_filename = popu_ref_filename, method='sp', use_memmap=False)
# fp.run_pca(pref_ref, pref_stu, popu_ref_filename = popu_ref_filename, method='ap', use_memmap=False)
# fp.run_pca(pref_ref, pref_stu, popu_ref_filename = popu_ref_filename, method='oadp', use_memmap=False)
fp.cmp_pcs(pref_stu, ['sp', 'ap', 'oadp'])

# fp.run_pca(pref_ref, pref_stu, popu_ref_k=5, method='sp', use_memmap=False)

# pcs_ref, pcs_stu_proj, pcs_stu_hdpca, pcs_stu_onl = fp.load_pcs('pcs')
# pcs_stu = pcs_stu_onl
# popu_ref = pd.read_table(popu_ref_filename)
# popu_stu_pred = fp.pred_popu_stu(pcs_ref, popu_ref, pcs_stu)
# fp.plot_pcs(pcs_ref, pcs_stu_proj, pcs_stu_hdpca, pcs_stu_onl, popu_ref, popu_stu_pred, pref_stu, marker_stu='.', alpha_stu=0.3)
