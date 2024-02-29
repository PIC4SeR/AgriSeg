#!/bin/bash
    
domains="tree_2 chard lettuce vineyard"
meth="None"

for target in $domains; do # Iterate on domains
    for i in 1; do # Multiple runs
        date
        echo "Training: meth=$meth, target=$target, id=$i"
        python3 main.py --target $target --id $i --config utils/config.yaml --method $meth --cuda 0 2>&1 | tee logs/teachers_tf_da.log
    done
done