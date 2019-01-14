#!/bin/bash
set -e

# Intersect the markers of two plink binary files
# The resulted two bed files have the same markers.
# Their order and reference alleles will also be the same.
# i.e. the two bim files will be identical

ref_prefname=$1
stu_prefname=$2

ref_basename=`basename ${ref_prefname}`
stu_basename=`basename ${stu_prefname}`
ref_snpscap_stu_prefname=${ref_prefname}_extractsnps_${stu_basename}
stu_snpscap_ref_prefname=${stu_prefname}_extractsnps_${ref_basename}
# stu_snpscap_ref_prefname_commrefal=${stu}_snpscap_${ref_basename}

if [ ! -f ${ref_snpscap_stu_prefname} ] || [ ! -f ${stu_snpscap_ref_prefname} ]; then
    cut -f2 ${stu_prefname}.bim > ${stu_prefname}.rs
    plink --bfile ${ref_prefname} --extract ${stu_prefname}.rs --out ${ref_snpscap_stu_prefname} --make-bed
    cut -f2 ${ref_snpscap_stu_prefname}.bim > ${ref_snpscap_stu_prefname}.rs
    plink --bfile ${stu_prefname} --extract ${ref_snpscap_stu_prefname}.rs --a1-allele ${ref_snpscap_stu_prefname}.bim 5 2 --out ${stu_snpscap_ref_prefname} --make-bed
    rm ${stu_prefname}.rs ${ref_snpscap_stu_prefname}.rs
    rm ${ref_snpscap_stu_prefname}.nosex ${stu_snpscap_ref_prefname}.nosex
    # cut -f2,5 ${ref_snpscap_stu_prefname}.bim > ${ref_snpscap_stu_prefname}.refal
    # plink --bfile ${stu_snpscap_ref_prefname} --reference-allele ${ref_snpscap_stu_prefname}.refal --out ${stu_snpscap_ref_prefname_commrefal} --make-bed
    cmp --silent ${ref_snpscap_stu_prefname}.bim ${stu_snpscap_ref_prefname}.bim || {>&2 echo "Error: Output bim files not identical"; echo; echo; exit 1; } # The last two echos are placeholders for output file names
    # rm ${stu_snpscap_ref_prefname}.bed ${stu_snpscap_ref_prefname}.bim ${stu_snpscap_ref_prefname}.fam
    echo "Intersection finished."
fi

echo 'Output files: '
echo ${ref_snpscap_stu_prefname}
echo ${stu_snpscap_ref_prefname}
