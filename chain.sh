#!/bin/bash

sim=$1
ver=$2

host=`hostname`
if [[ "${host:0:3}" == "nyx" || "${host:0:4}" == "flux" ]]; then
    echo "Loading modules for flux..."
    module load python-dev 
    module load numpy-dev 
    module load mkl/11.3.3 # Dependency for armadillo
    module load gcc/5.4.0 # Dependency for armadillo
    module load armadillo
    module load R
    export LD_LIBRARY_PATH=/home/daiweiz/gsl/lib:$LD_LIBRARY_PATH;
    echo "Done."
elif [ ${host} == "david-XPS-13-9343" ]; then
    echo "Using Python virtualenv for local."
    source virtualenvwrapper.sh
    workon research
    python --version
fi

if [ ${sim} == "1" ]; then
    pref=$3
    p=$4
    n=$5
    m=$6
    k=$7
    s=$8
    mig=$9
    bash simGeno.sh ${ver} ${pref} ${p} ${n} ${m} ${k} ${s} ${mig}
    geno_file=../data/${pref}/${pref}_${p}_${n}_${k}_${s}_${mig}_0.geno
    study_file=../data/${pref}/${pref}_${p}_${m}_${k}_${s}_${mig}_1.geno
    out_pref=../data/${pref}/${pref}_${p}_${n}_${m}_${k}_${s}_${mig}.${ver}
elif [ ${sim} == "0" ]; then
    echo "Using existing .geno files."
    refd=$3
    refn=$4
    stud=$5
    stun=$6
    outid=$7
    outd=${refd}_${stud}_${outid}
    outn=${refn}_${stun}
    study_file=../data/${stud}/${stun}.geno
    geno_file=../data/${refd}/${refn}.geno
    out_pref=../data/${outd}/${outn}.${ver}
else
    echo "Invalid input for sim."
    exit 1
fi

if [ -f ${out_pref}.info ]; then
    rm ${out_pref}.info
fi


date
bash runTrace.sh ${ver} ${geno_file} ${study_file} ${out_pref}

date
Rscript runHdpca.R ${out_pref} hdpca

date
Rscript runHdpca.R ${out_pref} hdpcaRand

date
Rscript accuracy.R ${out_pref}



# ver=$1
# pref=$2
# p=$3
# # n and m must be a multiple of K^2, where K is the grid size in the ggs program
# n=$4
# m=$5
# k=$6
# s=$7
# mig=$8
# let "nplusm = n + m"
# let "nplusmHigh = nplusm"
# # Not using unbalanced for now. Kind of buggy right now.
# # let "h = k * k"
# # let "nplusmHigh = nplusm * (h * s) / (s + h) -1"



# date
# echo $ver $pref $p $n $m $k $s $mig
# mkdir ../data/${pref}

# GENO_FILE=../${pref}/${pref}_${p}_${n}_${k}_${s}_${mig}_0.geno
# STUDY_FILE=../${pref}/${pref}_${p}_${m}_${k}_${s}_${mig}_1.geno
# ggs_file=../${pref}/${pref}_${p}_${nplusmHigh}_${k}_${mig}.ggs
# dup_file=../${pref}/${pref}_${p}_${nplusmHigh}_${k}_${s}_${mig}.geno.tr
# geno_both_file=../${pref}/${pref}_${p}_${nplusmHigh}_${k}_${s}_${mig}.geno
# site_both_file=../${pref}/${pref}_${p}_${nplusmHigh}_${k}_${s}_${mig}.site

# if [[ -f ${GENO_FILE} || -f ${STUDY_FILE} ]] ; then
#     echo "Using existing GENO_FILE: ${GENO_FILE}"
#     echo "Using existing STUDY_FILE: ${STUDY_FILE}"
# else
#     echo "Generating data with GGS..."
#     bash runGgs.sh ${pref} ${p} ${nplusmHigh} ${k} ${mig}
#     date
#     bash ggs2trace.sh ${pref} ${p} ${nplusmHigh} ${k} ${s} ${mig}
#     date
#     bash getHalfRows.sh ${pref} ${p} ${n} ${m} ${k} ${s} ${mig}
#     date
#     echo "Done."
#     # TODO: Correct the code below. It's not working
#     # echo "Removing intermediate files..."
#     # rm ${ggs_file} ${dup_file} ${geno_both_file} ${site_both_file}
#     # echo "Done."
# fi

# bash runTrace.sh ${ver} ${pref} ${p} ${n} ${m} ${k} ${s} ${mig}
# date
# Rscript runHdpca.R ${ver} hdpca ${pref} ${p} ${n} ${m} ${k} ${s} ${mig}
# date
# Rscript runHdpca.R ${ver} hdpcaRand ${pref} ${p} ${n} ${m} ${k} ${s} ${mig}
# date

# pdf=../data/${pref}/${pref}_${p}_${n}_${m}_${k}_${s}_${mig}.pdf
# rm ${pdf}
# Rscript plot.R ${ver} ${pref} ${p} ${n} ${m} ${k} ${s} ${mig}
# # evince ${pdf}



# # Unused
# # ./getWeightedCols ${pref} ${p} ${n} ${m}
# # rm ../data/${pref}/${pref}_${p}_${nplusm}.ggs
