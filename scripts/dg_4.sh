#!/bin/bash
date
python3 main.py --id 0 --cfg cfg/cfg_3.yaml --method KD --cuda 6 >> logs/bench.log

# ERM Teachers
#for target in $domains; do # Iterate on domains
#    for i in 1; do # Multiple runs
#        date
#        echo "Training: meth=$meth, target=$target, id=$i"
#        python3 main.py --target $target --id $i --cfg utils/cfg.yaml --method $meth --erm_teacher
#    done
#done