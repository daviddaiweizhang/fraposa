#!/bin/bash
# Split a large bed files into smaller files by individuals

set -e

filepref=$1
n_chunks=$2
i=$3

chunk_len_suff=4
chunk_midf=nchunks${n_chunks}
dir=${filepref}_${chunk_midf}
basepref=`basename ${filepref}`
chunk_filepref=${dir}/${basepref}_${chunk_midf}_`printf "%0${chunk_len_suff}d\n" ${i}`
mkdir -p ${dir}

# Split .fam file
echo ${filepref}
echo ${chunk_filepref}
split -d -n l/${i}/${n_chunks} -a ${chunk_len_suff} ${filepref}.fam > ${chunk_filepref}

plink --bfile ${filepref} --keep ${chunk_filepref} --keep-allele-order --out ${chunk_filepref} --make-bed
cmp --silent ${chunk_filepref}.bim ${filepref}.bim || {>&2 echo "Error: Output bim files not identical"; echo; echo; exit 1; } # The last two echos are placeholders for output file names
rm ${chunk_filepref}
# Return the list of splitted files
echo ${chunk_filepref}

# TODO: check concat(all chunck .fam) == .fam
