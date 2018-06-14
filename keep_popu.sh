#!/bin/bash

set -e

prefname_ref=$1
prefname_stu=$2
popu_name_this=$3
method=ap

prefname_ref_this=${prefname_ref}_${popu_name_this}
prefname_stu_this=${prefname_stu}_pred_${popu_name_this}

if [ ! -f ${prefname_stu_this}.bed ]; then
    grep ${popu_name_this} ${prefname_stu}_pred_${method}.popu > ${prefname_stu_this}.popu
    plink --bfile ${prefname_stu} --keep ${prefname_stu_this}.popu --keep-allele-order --out ${prefname_stu_this} --make-bed
fi

if [ ! -f ${prefname_ref_this}.bed ]; then
    paste ${prefname_ref}_sub.popu ${prefname_ref}.popu | grep ${popu_name_this} | cut -f1-3 | sed -e 's/ /\t/g' > ${prefname_ref_this}.popu
    plink --bfile ${prefname_ref} --keep ${prefname_ref_this}.popu --keep-allele-order --out ${prefname_ref_this} --make-bed
fi

echo ${prefname_ref_this}
echo ${prefname_stu_this}
