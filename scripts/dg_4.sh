#!/bin/bash
    
domains="vineyard"
meth="KD"
name="ema_2"

for i in 2 3 4 5; do # Multiple runs
    for target in $domains; do # Iterate on domains
        date
        echo "Training: name=$name, meth=$meth, target=$target, id=$i"
        python3 main.py --target $target --id $i --config cfg/config_1.yaml --name $name --method $meth --cuda 6 2>&1 | tee logs/${name}_$i.log
    done
done

# ERM Teachers
#for target in $domains; do # Iterate on domains
#    for i in 1; do # Multiple runs
#        date
#        echo "Training: meth=$meth, target=$target, id=$i"
#        python3 main.py --target $target --id $i --config utils/config.yaml --method $meth --erm_teacher
#    done
#done