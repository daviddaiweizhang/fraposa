#!/bin/bash

# Run the ggs software with the given paramters

# Prefix of the output file
PREF=$1
# The number of SNPs
P=$2
# The number of individuals
N=$3
K=$4
M=$5

# See ggs manual for these parameters
L=100
# Make sure the remainder is zero. No sanity check here.
let "C = ${N} / (${K} * ${K}) * 2"
let G=${P}/${L}

cd ../data/ggs
./ggs  -K ${K} -c ${C} -M ${M} -G ${G} -L ${L} -o ../${PREF}/${PREF}_${P}_${N}_${K}_${M}.ggs > ggs.log 2>&1
