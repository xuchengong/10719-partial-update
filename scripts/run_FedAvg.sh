#!/bin/bash

cd ..
cd system


seed=0

PIDS=()
dev=0

for NC in 10 20 30 40 50 60 70 80 90 100
do
    TIME=$(date +'%y-%m-%d %T')\ NC=${NC}
    mkdir -p "../results/$TIME"
    CUDA_VISIBLE_DEVICES=0 nohup python -u main.py -seeds $seed -ls 1 -lbs 16 -eg 5 -lr 0.01 -nc $NC -nb 62 -data femnist -m even_cnn -algo FedAvg -gr 1500 -go first -ab True -fn "$TIME" 2>&1 |tee "../results/$TIME/baseline_${seed}_${NC}.out" &
    PIDS+=($!)
done
wait "${PIDS[@]}"