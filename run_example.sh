#!/bin/bash
set -e

origin=umich.edu/~daiweiz/fraposa
refpref_raw=data/thousandGenomes
stupref_raw=data/exampleStudySamples
refpref=data/thousandGenomes_comm
stupref=data/exampleStudySamples_comm

for suff in bed bim fam popu; do
    mkdir -p `dirname $refpref_raw`
    wget -O $refpref_raw.$suff $origin/$refpref_raw.$suff
done

for suff in bed bim fam; do
    mkdir -p `dirname $stupref_raw`
    wget -O $stupref_raw.$suff $origin/$stupref_raw.$suff
done

./commvar.sh $refpref_raw $stupref_raw $refpref $stupref
./fraposa_runner.py --stu_filepref $stupref $refpref
./predstupopu.py $refpref $stupref
./plotpcs.py $refpref $stupref
