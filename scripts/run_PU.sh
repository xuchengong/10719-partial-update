#!/bin/bash

cd ..
cd system


seed=0

PIDS=()
dev=0

for CR in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
do
    TIME=$(date +'%y-%m-%d %T')\ CR3=${CR}
    mkdir -p "../results/$TIME"
    CUDA_VISIBLE_DEVICES=0 nohup python -u main.py -seeds $seed -cr $CR -ls 1 -lbs 16 -eg 5 -lr 0.01 -nc 100 -nb 62 -data femnist -m even_cnn -algo ServerPU -gr 1500 -go first -ab True -fn "$TIME" 2>&1 |tee "../results/$TIME/baseline_${seed}_${CR}.out" &
    PIDS+=($!)
done
wait "${PIDS[@]}"