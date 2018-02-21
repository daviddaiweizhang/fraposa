#!/bin/bash

ver=$1
pref=$2
p=$3
# n and m must be a multiple of K^2, where K is the grid size in the ggs program
n=$4
m=$5
k=$6
s=$7
mig=$8
let "nplusm = n + m"
let "nplusmHigh = nplusm"
# Not using unbalanced for now. Kind of buggy right now.
# let "h = k * k"

echo "Data simulation for input files (.geno and .site) for TRACE..."

geno_file=../data/${pref}/${pref}_${p}_${n}_${k}_${s}_${mig}_0.geno
study_file=../data/${pref}/${pref}_${p}_${m}_${k}_${s}_${mig}_1.geno
out_prefix=../data/${pref}/${pref}_${p}_${n}_${m}_${k}_${s}_${mig}.${ver}
ggs_file=../data/${pref}/${pref}_${p}_${nplusmHigh}_${k}_${mig}.ggs
geno_both_file=../data/${pref}/${pref}_${p}_${nplusmHigh}_${k}_${s}_${mig}.geno
dup_file=${geno_both_file}.tr

if [[ -f ${geno_file} && -f ${study_file} ]]; then
    echo "Using existing data files."
else
    echo "Generating data with GGS..."
    mkdir -p ../data/${pref}
    bash runGgs.sh ${pref} ${p} ${nplusmHigh} ${k} ${mig}
    date
    bash ggs2trace.sh ${pref} ${p} ${nplusmHigh} ${k} ${s} ${mig}
    date
    bash getHalfRows.sh ${pref} ${p} ${n} ${m} ${k} ${s} ${mig}
    date
    echo "Done."
    echo "Removing intermediate files..."
    rm ${ggs_file} ${dup_file} ${geno_both_file} 
    echo "Done."
fi

echo "Finished!"
