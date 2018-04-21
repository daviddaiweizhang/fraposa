import pandas as pd
import frugalpca as fp

ref_pref = '../data/kgn/kgn_snps_ukb'
stu_pref = '../data/ukb/ukb_snps_kgn'
popu_ref_filename = '../data/kgn/kgn_orphans.popu'

log = fp.create_logger(stu_pref)

fp.test_online_svd_procrust()

fp.run_pca(ref_pref, stu_pref, popu_ref_filename)

# pcs_ref, pcs_stu_proj, pcs_stu_hdpca, pcs_stu_onl = fp.load_pcs('pcs')
# popu_ref = pd.read_table(popu_ref_filename)[popu_col_name]
# popu_stu_pred = fp. pred_popu_stu(pcs_ref, popu_ref, pcs_stu_onl)
# fp.plot_pcs(pcs_ref, pcs_stu_proj, pcs_stu_hdpca, pcs_stu_onl, popu_ref, popu_stu_pred, stu_pref, marker_stu='.', alpha_stu=0.3)
