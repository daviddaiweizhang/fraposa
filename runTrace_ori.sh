#!/bin/bash

ver=$1
geno_file=$2
study_file=$3
out_pref=$4

DIM=4
DIM_HIGH=20

echo "Running TRACE..."

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

trace -p trace.conf
