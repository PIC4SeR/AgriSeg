#!/bin/bash
    
domains="tree_2 chard lettuce vineyard"
meth="KD"

for target in $domains; do # Iterate on domains
    for i in 1 2 3 4 5; do # Multiple runs
        date
        echo "Training: meth=$meth, target=$target, id=$i"
        python3 main.py --target $target --id $i --config utils/config_3.yaml --method $meth --cuda 2 2>&1 | tee logs/KD_geom_wcta.log
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