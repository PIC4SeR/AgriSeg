#!/bin/bash
    
domains="tree_2 chard lettuce vineyard"
meth="None"
name="eval"

# for target in $domains; do # Iterate on domains
#     for i in 1; do # Multiple runs
#         date
#         echo "Training: meth=$meth, target=$target, id=$i"
#         python3 main.py --target $target --id $i --config cfg/config_1.yaml --method $meth --cuda 7 --name $name 2>&1 | tee logs/$name.log
#     done
# done

# Test Teachers
for target in $domains; do # Iterate on domains
    for i in 1; do # Multiple runs
        date
        echo "Training: meth=$meth, target=$target, id=$i"
        python3 main.py --target $target --id $i --config cfg/config_2.yaml --method $meth --cuda 7 --test --weights bin/teachers/tf_wcta/teacher_${target}.h5 # 2>&1 | tee logs/teachers_tf.log
    done
done