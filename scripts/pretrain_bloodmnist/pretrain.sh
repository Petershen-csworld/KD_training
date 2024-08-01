#!/bin/bash
seed=3407
dataset="BloodMNIST"
models=("resnet18" "resnet20" "resnet56" "wrn_40_2" "wrn_16_2")
counter=1

for model_name in "${models[@]}"; do
  log_file="out/output_note${counter}.log"
  nohup python3 ./teacher_orig_ds.py --seed=${seed} --model_name=${model_name}\
    --dataset=${dataset} --gpu=${counter} > ${log_file} 2>&1 &
  ((counter++))
done