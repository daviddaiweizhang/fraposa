#!/bin/bash

for n in `seq 1000 500 3000`; do
    bash simGeno.sh 1 ggsim 100000 ${n} 200 2 1 100
done
