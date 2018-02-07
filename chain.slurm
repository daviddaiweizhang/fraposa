#!/bin/sh
#SBATCH --mail-type=ALL
#SBATCH --mail-user=daiweiz@umich.edu
#SBATCH --job-name=chain_rand
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --mem=32000
#SBATCH -o chain_rand.slurm.log

#i=SLURM_ARRAY_TASK_ID

date
./chain_simp rand kgn kgn_allChr_ukb_orphans ukb ukb_1k
date
