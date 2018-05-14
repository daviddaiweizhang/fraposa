#!/bin/bash

set -e

ref=$1
stu=$2

pref_ref_commsnps=${ref}_snps_comm
pref_stu_commsnps=${stu}_refal_diff
pref_stu_commsnps_commrefal=${stu}_snps_comm

cut -f2 ${ref}.bim > ${ref}.rs
cut -f2 ${stu}.bim > ${stu}.rs
plink --noweb --bfile ${ref} --extract ${stu}.rs --snps-only --out ${pref_ref_commsnps} --make-bed
plink --noweb --bfile ${stu} --extract ${ref}.rs --snps-only --out ${pref_stu_commsnps} --make-bed
cut -f2,5 ${pref_ref_commsnps}.bim > ${pref_ref_commsnps}.refal
plink --noweb --bfile ${pref_stu_commsnps} --reference-allele ${pref_ref_commsnps}.refal --out ${pref_stu_commsnps_commrefal} --make-bed
cmp --silent ${pref_ref_commsnps}.bim ${pref_stu_commsnps_commrefal}.bim || {>&2 echo "Error: Output bim files not identical"; echo; echo; exit 1; } # The last two echos are placeholders for output file names
rm ${pref_stu_commsnps}
echo "Intersection finished."
echo ${pref_ref_commsnps}
echo ${pref_stu_commsnps_commrefal}
