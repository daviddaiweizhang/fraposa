#!/bin/bash

set -e

ref=$1
stu=$2
tmp_dirname=$3

ref_basename=`basename ${ref}`
stu_basename=`basename ${stu}`
pref_ref_commsnps=${ref}_snpscap_${stu_basename}
pref_stu_commsnps=${stu}_snpscap_${ref_basename}_refaldiff
pref_stu_commsnps_commrefal=${stu}_snpscap_${ref_basename}

if [ ! -f ${pref_ref_commsnps} ] || [ ! -f ${pref_stu_commsnps_commrefal} ]; then
    cut -f2 ${ref}.bim > ${ref}.rs
    cut -f2 ${stu}.bim > ${stu}.rs
    plink --bfile ${ref} --extract ${stu}.rs --snps-only --out ${pref_ref_commsnps} --make-bed
    plink --bfile ${stu} --extract ${ref}.rs --snps-only --out ${pref_stu_commsnps} --make-bed
    cut -f2,5 ${pref_ref_commsnps}.bim > ${pref_ref_commsnps}.refal
    plink --bfile ${pref_stu_commsnps} --reference-allele ${pref_ref_commsnps}.refal --out ${pref_stu_commsnps_commrefal} --make-bed
    cmp --silent ${pref_ref_commsnps}.bim ${pref_stu_commsnps_commrefal}.bim || {>&2 echo "Error: Output bim files not identical"; echo; echo; exit 1; } # The last two echos are placeholders for output file names
    rm ${pref_stu_commsnps}.bed ${pref_stu_commsnps}.bim ${pref_stu_commsnps}.fam
    echo "Intersection finished."
fi

echo ${pref_ref_commsnps}
echo ${pref_stu_commsnps_commrefal}
