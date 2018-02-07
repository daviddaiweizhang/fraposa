#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --job-name=mChg_rand
#SBATCH --time=0-02:00:00
#SBATCH --nodes=1
#SBATCH --mem=32000
#SBATCH --output=slurm_log/mChg_rand.%a.%A.log
#SBATCH --array=1000-3000:500

m=${SLURM_ARRAY_TASK_ID}
id=${SLURM_ARRAY_JOB_ID}
# id=`date +%s`

date
./chain rand ${id} 100000 600 $m 2 1 100
date
