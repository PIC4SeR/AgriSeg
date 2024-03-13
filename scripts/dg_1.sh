#!/bin/bash
    
domains="tree_2 chard lettuce vineyard"
meth="KD"
name="good"

for i in 3 4 5; do # Multiple runs
    for target in $domains; do # Iterate on domains
        date
        echo "Training: meth=$meth, target=$target, id=$i"
        python3 main.py --target $target --id $i --cfg cfg/cfg_2.yaml --method $meth --cuda 5 --name $name >> logs/${name}_$i.log
    done
done

# ERM Teachers
#for target in $domains; do # Iterate on domains
#    for i in 1; do # Multiple runs
#        date
#        echo "Training: meth=$meth, target=$target, id=$i"
#        python3 main.py --target $target --id $i --cfg cfg/cfg.yaml --method $meth --erm_teacher
#    done
#done