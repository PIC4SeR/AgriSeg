#!/bin/bash

domains="misc"
meth="KD"
name="ablation_2"

for i in 1 2 3; do # Multiple runs
    for target in $domains; do # Iterate on domains
        for a in 0.1; do
            for t in 1 3; do
                date
                echo "Training: meth=$meth, target=$target, id=$i, $a, $t"
                python3 main.py --target $target --id $i --cfg cfg/cfg_2.yaml --method $meth --cuda 0 --alpha $a --temperature $t --name $name >> logs/${name}_${target}.log
            done
        done
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