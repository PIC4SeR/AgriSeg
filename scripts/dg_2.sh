#!/bin/bash
    
domains="lettuce vineyard"
meth="KD"

for i in 6 7 8; do # Multiple runs
    for target in $domains; do # Iterate on domains
        date
        echo "Training: meth=$meth, target=$target, id=$i"
        python3 main.py --target $target --id $i --config cfg/config_2.yaml --method $meth --cuda 2 2>&1 | tee logs/KD_geom_geom_wcta_last.log
    done
done

# ERM Teachers
#for target in $domains; do # Iterate on domains
#    for i in 1; do # Multiple runs
#        date
#        echo "Training: meth=$meth, target=$target, id=$i"
#        python3 main.py --target $target --id $i --config cfg/config.yaml --method $meth --erm_teacher
#    done
#done