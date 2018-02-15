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
let "h = k * k"
# Not using unbalanced for now. Kind of buggy
# let "nplusmHigh = nplusm * (h * s) / (s + h) -1"
let "nplusmHigh = nplusm"

module load python-dev 
module load numpy-dev 
# module load gsl # Using gsl installed on home
module load mkl/11.3.3 # Dependency for armadillo
module load gcc/5.4.0 # Dependency for armadillo
module load armadillo
module load R
module list


date
echo $ver $pref $p $n $m $k $s $mig
mkdir ../data/${pref}

GENO_FILE=../${pref}/${pref}_${p}_${n}_${k}_${s}_${mig}_0.geno
STUDY_FILE=../${pref}/${pref}_${p}_${m}_${k}_${s}_${mig}_1.geno
ggs_file=../${pref}/${pref}_${p}_${nplusmHigh}_${k}_${mig}.ggs
dup_file=../${pref}/${pref}_${p}_${nplusmHigh}_${k}_${s}_${mig}.geno.tr
geno_both_file=../${pref}/${pref}_${p}_${nplusmHigh}_${k}_${s}_${mig}.geno
site_both_file=../${pref}/${pref}_${p}_${nplusmHigh}_${k}_${s}_${mig}.site

if [[ -f ${GENO_FILE} || -f ${STUDY_FILE} ]] ; then
    echo "Using existing GENO_FILE: ${GENO_FILE}"
    echo "Using existing STUDY_FILE: ${STUDY_FILE}"
else
    echo "Generating data with GGS..."
    bash runGgs.sh ${pref} ${p} ${nplusmHigh} ${k} ${mig}
    date
    bash ggs2trace.sh ${pref} ${p} ${nplusmHigh} ${k} ${s} ${mig}
    date
    bash getHalfRows.sh ${pref} ${p} ${n} ${m} ${k} ${s} ${mig}
    date
    echo "Done."
    echo "Removing intermediate files..."
    rm ${ggs_file} ${dup_file} ${geno_both_file} ${site_both_file}
    echo "Done."
fi

bash runTrace.sh ${ver} ${pref} ${p} ${n} ${m} ${k} ${s} ${mig}
date
Rscript runHdpca.R ${ver} hdpca ${pref} ${p} ${n} ${m} ${k} ${s} ${mig}
date
Rscript runHdpca.R ${ver} hdpcaRand ${pref} ${p} ${n} ${m} ${k} ${s} ${mig}
date

pdf=../data/${pref}/${pref}_${p}_${n}_${m}_${k}_${s}_${mig}.pdf
rm ${pdf}
Rscript plot.R ${ver} ${pref} ${p} ${n} ${m} ${k} ${s} ${mig}
# evince ${pdf}



# Unused
# ./getWeightedCols ${pref} ${p} ${n} ${m}
# rm ../data/${pref}/${pref}_${p}_${nplusm}.ggs
