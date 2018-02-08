#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --job-name=nChg_comb
#SBATCH --time=0-02:00:00
#SBATCH --nodes=1
#SBATCH --mem=32000
#SBATCH --output=slurm_log/nChg_comb.%a.%A.log
#SBATCH --array=1000-3000:500

n=${SLURM_ARRAY_TASK_ID}
id=${SLURM_ARRAY_JOB_ID}
# id=`date +%s`

date
./chain comb ${id} 100000 $n 200 2 1 100
date
