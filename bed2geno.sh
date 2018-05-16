#!/bin/bash

bed_filepref=$1

plink --bfile ${bed_filepref} --recode vcf --out ${bed_filepref}
vcf2geno --inVcf ${bed_filepref}.vcf --out ${bed_filepref}
