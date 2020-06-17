#!/bin/bash
set -e

# Extract the variants that are commen to both binary PLINK files

if_one_prefname=$1
if_two_prefname=$2
of_one_prefname=$3
of_two_prefname=$4

if_one_basename=`basename ${if_one_prefname}`
if_two_basename=`basename ${if_two_prefname}`
of_one_basename=`basename ${of_one_prefname}`
of_two_basename=`basename ${of_two_prefname}`

cut -f2 ${if_two_prefname}.bim > tmp.rs
plink --bfile ${if_one_prefname} --extract tmp.rs --out ${of_one_prefname} --make-bed
cut -f2 ${of_one_prefname}.bim > tmp.rs
plink --bfile ${if_two_prefname} --extract tmp.rs --a1-allele ${of_one_prefname}.bim 5 2 --out ${of_two_prefname} --make-bed
rm tmp.rs

if [ -s "${if_one_prefname}.popu" ]; then
    cp ${if_one_prefname}.popu ${of_one_prefname}.popu
fi
if [ -s "${if_two_prefname}.popu" ]; then
    cp ${if_two_prefname}.popu ${of_two_prefname}.popu
fi
