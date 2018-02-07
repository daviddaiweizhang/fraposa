#!/bin/bash

ver=$1
refd=$2
refn=$3
stud=$4
stun=$5
outd=${stud}
outn=${refn}_${stun}

name=trace.${ver}
runTrace_args="${ver} ${refd} ${refn} ${stud} ${stun} ${outd} ${outn}"

date

module load python-dev
module load numpy-dev
# module load gsl # Using gsl installed on home
module load mkl/11.3.3 # Dependency for armadillo
module load gcc/5.4.0 # Dependency for armadillo
module load armadillo
module load R
module list


date
./runTrace_simp ${runTrace_args}
date
p=`grep "loci shared by STUDY_FILE and GENO_FILE" ../data/${outd}/${outn}.${ver}.log | cut -f 2 -d " "`
n=`grep "individuals are detected in the GENO_FILE" ../data/${outd}/${outn}.${ver}.log | cut -f 1 -d " "`
m=`grep "individuals are detected in the STUDY_FILE" ../data/${outd}/${outn}.${ver}.log | cut -f 1 -d " "`
runHdpca_args="${ver} ${outd} ${outn} $p $n"
date
Rscript runHdpca_simp.R ${runHdpca_args}
date
Rscript plot_simp.R ${ver} ${stud} ${outn} $p $n $m
date

# pdf=../data/${pref}/${pref}_${p}_${n}_${m}_${k}_${s}_${mig}.pdf
# rm ${pdf}
# Rscript plot.R ${ver} ${pref} ${p} ${n} ${m} ${k} ${s} ${mig}
# evince ${pdf}



# Unused
# ./getWeightedCols ${pref} ${p} ${n} ${m}
# rm ../data/${pref}/${pref}_${p}_${nplusm}.ggs
