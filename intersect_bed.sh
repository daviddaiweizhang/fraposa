#!/bin/bash

ref=$1
stu=$2
tmpdir=$3

cut -f2 ${ref}.bim > ${ref}.rs
cut -f2 ${stu}.bim > ${stu}.rs
plink --noweb --bfile ${ref} --extract ${stu}.rs --snps-only --out ${tmpdir}/ref_commsnpsrefal --make-bed
plink --noweb --bfile ${stu} --extract ${ref}.rs --snps-only --out ${tmpdir}/stu_commsnps --make-bed
cut -f2,5 ${tmpdir}/ref_commsnpsrefal.bim > ${tmpdir}/ref_commsnpsrefal.refal
plink --noweb --bfile ${tmpdir}/stu_commsnps --reference-allele ${tmpdir}/ref_commsnpsrefal.refal --out ${tmpdir}/stu_commsnpsrefal --make-bed
cmp --silent ${tmpdir}/ref_commsnpsrefal.bim ${tmpdir}/stu_commsnpsrefal.bim || {>&2 echo "Error: Output bim files not identical"; echo; echo; exit 1; } # The last two echos are placeholders for output file names
echo "Intersection finished."
echo ${tmpdir}/ref_commsnpsrefal
echo ${tmpdir}/stu_commsnpsrefal
