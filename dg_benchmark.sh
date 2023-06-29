#!/bin/bash
    
domains="vineyard_real_new"

for target in $domains; do # Iterate on domains
    for i in 1 2 3 4 5; do # Multiple runs
        date
        echo "Training: meth=$meth, target=$target"
        python3 main.py --target $target --id $i --config utils/config.yaml >> logs/ISW.txt
    done
done
