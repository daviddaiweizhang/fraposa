import frugalpca as fp

# Data:
# Reference and study data should
# 1. be in the plink binary format (blah.bed + blah.bim + blah.fam)
# 2. have the same markers. Their orders and reference alleles also need to be the same.
#    (i.e. reference_data.bim and study_data.bim are identical)
#    If not, run intersect_bed.sh before running the example below.

# Python:
# Python version should be 3.x
# You may need to install the modules needed

# The reference panel. Ussing 1000 genomes here.
ref_filepref = '../data/kgn/kgn_bial_orphans_snps_ukb_snpscap_ukb'
# 500 samples randomly selected from the UK Biobank data 
stu_filepref = '../data/ukb/ukb_snpscap_kgn_bial_orphans_5c'

# See the definition of run_pca() for all the parameters that you can set
fp.run_pca(ref_filepref, stu_filepref,
           method='ap', # choose 'sp' (simple projection), 'ap' (Rounak's hdpca), or 'oadp' (online svd)
           hdpca_n_spikes=4, # For method=ap only. The number of distant spikes. If not set, hdpca will predict it.
           plot_results=True, # If you want the PC scores to be plotted
           plot_size=(12,4) # Not necessary to set. Just to make the the plot look a little nicer
)
