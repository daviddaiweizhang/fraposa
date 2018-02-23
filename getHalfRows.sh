#!/bin/bash

# Get n random rows from inFile of m lines and save them to outFile
# OR
# Get every other row from inFile of m lines and save them to outFile
# See the python file to find which version is used

pref=$1
p=$2
n=$3
m=$4
k=$5
s=$6
mig=$7
let "nplusm = n + m"

name=../data/${pref}/${pref}_${p}_${n}_${m}_${k}_${s}_${mig}
python run_getHalfRows.py \
	../data/${pref}/${pref}_${p}_${nplusm}_${k}_${s}_${mig}.geno \
	${n} \
	${nplusm} \
	../data/${pref}/${pref}_${p}_${n}_${k}_${s}_${mig}_0.geno \
	../data/${pref}/${pref}_${p}_${m}_${k}_${s}_${mig}_1.geno

cp ../data/ggs/${p}.site ../data/${pref}/${pref}_${p}_${n}_${k}_${s}_${mig}_0.site
cp ../data/ggs/${p}.site ../data/${pref}/${pref}_${p}_${m}_${k}_${s}_${mig}_1.site
