#!/bin/bash

python3 main.py --target tree_2 --id 0 --config cfg/config_1.yaml --method KD 2>&1 | tee trials/tree_2.log

python3 main.py --target lettuce --id 0 --config cfg/config_1.yaml --method KD 2>&1 | tee trials/da.log
