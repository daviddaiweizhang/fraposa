#!/bin/bash
# Split a large bed files into smaller files by individuals

set -e

bed_file_pref=$1
tmp_dir=$2
chunk_name_pref=$3

chunk_n_lines=5000
chunk_len_suff=4
chunk_file_pref=${tmp_dir}/${chunk_name_pref}
fam_file=${bed_file_pref}.fam

split --lines=${chunk_n_lines} --suffix-length=${chunk_len_suff} -d ${fam_file} ${chunk_file_pref}
for chunk_file in ${chunk_file_pref}*; do
    # plink --bfile ${bed_file_pref} --keep ${chunk_file} --out ${chunk_file} --make-bed
    plink --bfile ${bed_file_pref} --out ${chunk_file} --make-bed
    date
done
