#!/bin/bash
    
domains="tree_2 chard lettuce vineyard"

for target in $domains; do # Iterate on domains
    for i in 1 2 3 4 5; do # Multiple runs
        date
        echo "Training: meth=$meth, target=$target"
        python3 main.py --target $target --id $i --config utils/config.yaml >> logs/new_KD.txt
    done
done
