#!/bin/sh
#SBATCH --mail-type=ALL
#SBATCH --mail-user=daiweiz@umich.edu
#SBATCH --job-name=frugalpca
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --mem=32000
#SBATCH -o frugalpca.log

#i=SLURM_ARRAY_TASK_ID

id=${SLURM_ARRAY_JOB_ID}

date
source ~/venv/bin/activate
python frugalpca.py
date
