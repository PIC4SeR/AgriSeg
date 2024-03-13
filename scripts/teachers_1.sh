#!/bin/bash
    
domains="tree_2 chard lettuce vineyard"
meth="None"
#name="eval"

# for target in $domains; do # Iterate on domains
#     for i in 1 2 3 4 5; do # Multiple runs
#         date
#         echo "Training: meth=$meth, target=$target, id=$i"
#         python3 main.py --target $target --id $i --cfg cfg/cfg_1.yaml --method $meth --cuda 0 --name sma 2>&1 | tee logs/sma.log
#     done
# done

# Test Teachers
for target in $domains; do # Iterate on domains
    for i in 1; do # Multiple runs
        date
        echo "Training: meth=$meth, target=$target, id=$i"
        python3 main.py --target $target --id $i --cfg cfg/cfg_5.yaml --method $meth --cuda 6 --test --weights bin/teachers/tf_geom_wcta/teacher_${target}.h5 # 2>&1 | tee logs/teachers_tf.log
    done
done