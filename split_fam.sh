#!/bin/bash
# Split a large bed files into smaller files by individuals

set -e

bed_filepref=$1
chunk_n_lines=$2

chunk_len_suff=4
chunk_midf=_chunksize${chunk_n_lines}_
chunk_filepref=${bed_filepref}${chunk_midf}

split -l ${chunk_n_lines} -a ${chunk_len_suff} -d ${bed_filepref}.fam ${chunk_filepref}
chunk_filepref_list=`ls ${chunk_filepref}* | egrep "^${chunk_filepref}[0-9]{${chunk_len_suff}}$"`
date
for chunk_filepref_this in ${chunk_filepref_list}; do
    plink --bfile ${bed_filepref} --keep ${chunk_filepref_this} --keep-allele-order --out ${chunk_filepref_this} --make-bed
    rm ${chunk_filepref_this}
    cmp --silent ${chunk_filepref_this}.bim ${bed_filepref}.bim || {>&2 echo "Error: Output bim files not identical"; echo; echo; exit 1; } # The last two echos are placeholders for output file names
    date
done
echo ${chunk_filepref_list}
