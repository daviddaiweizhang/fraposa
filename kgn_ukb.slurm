#!/bin/sh
#SBATCH --mail-type=ALL
#SBATCH --mail-user=daiweiz@umich.edu
#SBATCH --job-name=kgn_ukb
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --mem=16000
#SBATCH -o kgn_ukb.slurm.log

#i=SLURM_ARRAY_TASK_ID

id=${SLURM_ARRAY_JOB_ID}

date
bash chain.sh 0 rand kgn kgn_allChr_ukb_orphans ukb ukb_1k ${id}
date
