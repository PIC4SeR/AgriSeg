#!/bin/bash
    
domains="tree_2 chard lettuce vineyard"
meth="KD"

for target in $domains; do # Iterate on domains
    for i in 1 2 3 4 5; do # Multiple runs
        date
        echo "Training: meth=$meth, target=$target, id=$i"
        python3 main.py --target $target --id $i --cfg utilscfg_2.yaml --method $meth >> logs/wl_kd_2.log
    done
done