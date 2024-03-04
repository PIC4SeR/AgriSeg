#!/bin/bash
    
domains="tree_2 chard lettuce vineyard"
meth="None"
name="wcta"

for target in $domains; do # Iterate on domains
    for i in 1; do # Multiple runs
        date
        echo "Training: meth=$meth, target=$target, id=$i"
        python3 main.py --target $target --id $i --config cfg/config_2.yaml --method $meth --cuda 7 --name $name 2>&1 | tee logs/$name.log
    done
done

# Test Teachers
# for target in $domains; do # Iterate on domains
#     for i in 1; do # Multiple runs
#         date
#         echo "Training: meth=$meth, target=$target, id=$i"
#         python3 main.py --target $target --id $i --config utils/config.yaml --method $meth --cuda 0 --test --weights bin/${target}_tf_style_wcta.h5 # 2>&1 | tee logs/teachers_tf.log
#     done
# done