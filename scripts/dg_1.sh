#!/bin/bash
    
domains="tree_2 chard lettuce vineyard"
meth="None"

for i in 1 2 3 4 5; do # Multiple runs
    for target in $domains; do # Iterate on domains
        date
        echo "Training: meth=$meth, target=$target, id=$i"
        python3 main.py --target $target --id $i --config cfg/config_2.yaml --method $meth --cuda 6 --name erm 2>&1 | tee logs/erm.log
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