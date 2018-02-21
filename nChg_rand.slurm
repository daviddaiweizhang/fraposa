#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --job-name=nChg_rand
#SBATCH --time=0-02:00:00
#SBATCH --nodes=1
#SBATCH --mem=16000
#SBATCH --output=slurm_log/nChg_rand.%a.%A.log
#SBATCH --array=1000-3000:500

n=${SLURM_ARRAY_TASK_ID}
id=${SLURM_ARRAY_JOB_ID}
pref=nChg_rand_${id}_${n}
# id=`date +%s`

date
bash chain.sh 1 rand ${pref} 100000 $n 200 2 1 100
date
