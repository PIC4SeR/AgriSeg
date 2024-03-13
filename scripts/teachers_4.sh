#!/bin/bash
    
domains="tree_2 chard lettuce vineyard"
meth="None"

for target in $domains; do # Iterate on domains
    for i in 1; do # Multiple runs
        date
        echo "Training: meth=$meth, target=$target, id=$i"
        python3 main.py --target $target --id $i --cfg cfg/cfg_4.yaml --method $meth --cuda 6 2>&1 | tee logs/new_style_wcta.log
    done
done