#!/bin/bash

domains="tree_2"
meth="KD"

for target in $domains; do # Iterate on domains
    for i in 5; do # Multiple runs
        date
        echo "Training: meth=$meth, target=$target, id=$i"
        python3 main.py --target $target --id $i --method $meth --config cfg/config_3.yaml --cuda 4 2>&1 | tee logs/soup.log
    done
done

# ERM Teachers
#for target in $domains; do # Iterate on domains
#    for i in 1; do # Multiple runs
#        date
#        echo "Training: meth=$meth, target=$target, id=$i"
#        python3 main.py --target $target --id $i --config utils/config.yaml --method $meth --erm_teacher
#    done
#done