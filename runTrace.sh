#!/bin/bash

ver=$1
geno_file=$2
study_file=$3
out_pref=$4
exe="trace.${ver}.o"

DIM=4
DIM_HIGH=20

echo "Running TRACE..."
echo Checking whether ${exe} needs to be recompiled...
make ${exe}
echo Done.

echo -e "hostname\t`hostname`" >> ${out_pref}.info

# For running trace in this directory
# Better than putting all the info as arguments. This way running gdb will be easier
echo "\
GENO_FILE ${geno_file}
STUDY_FILE ${study_file}
OUT_PREFIX ${out_pref}
DIM        	${DIM}
DIM_HIGH	${DIM_HIGH}
" > trace.conf

./${exe} -p trace.conf

# cd ../data/laser
# echo "Entered laser dir"
# For running trace in ../data/laser
# echo "\
# STUDY_FILE	../${pref}/${pref}_${p}_${m}_${k}_${s}_${mig}_1.geno
# GENO_FILE	../${pref}/${pref}_${p}_${n}_${k}_${s}_${mig}_0.geno
# OUT_PREFIX	../${pref}/${pref}_${p}_${n}_${m}_${k}_${s}_${mig}.${ver}
# DIM        	${DIM}
# DIM_HIGH	${DIM_HIGH}
# " > trace.conf
