#!/bin/bash
domains="vegann"
meth="None"
name="baseline"

for i in 1 2 3 4 5; do # Multiple runs
    for target in $domains; do # Iterate on domains
        date
        echo "Training: meth=$meth, target=$target, id=$i"
        python3 main.py --target $target --id $i --cfg cfg/cfg_4.yaml --method $meth --cuda 5 --name $name >> logs/${name}_$i.log
    done
done

# ERM Teachers
#for target in $domains; do # Iterate on domains
#    for i in 1; do # Multiple runs
#        date
#        echo "Training: meth=$meth, target=$target, id=$i"
#        python3 main.py --target $target --id $i --cfg utils/cfg.yaml --method $meth --erm_teacher
#    done
#done