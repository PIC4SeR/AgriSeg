#!/bin/bash
    
domains="vite_reale"
id=1

for meth in "IN"; do
    for target in $domains; do # Iterate on domains
        for i in 1; do # Multiple runs
            date
            echo "Training: meth=$meth, target=$target"

            python3 main.py --target $target --id $i --method $meth --config utils/config.yaml >> logs/IN.txt

        done
    done
done