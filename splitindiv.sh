#!/bin/bash
# Split a large bed files into smaller files by individuals

set -e

filepref=$1
n_chunks=$2
i=$3
chunk_filepref=$4

chunk_len_suff=4
basepref=`basename ${filepref}`

# Split .fam file
split -d -n l/${i}/${n_chunks} -a ${chunk_len_suff} ${filepref}.fam > ${chunk_filepref}
plink --bfile ${filepref} --keep ${chunk_filepref} --keep-allele-order --out ${chunk_filepref} --make-bed
cmp --silent ${chunk_filepref}.bim ${filepref}.bim || {>&2 echo "Error: Output bim files not identical"; echo; echo; exit 1; } # The last two echos are placeholders for output file names
rm -f ${chunk_filepref}
echo ${chunk_filepref}

# TODO: check concat(all chunck .fam) == .fam
