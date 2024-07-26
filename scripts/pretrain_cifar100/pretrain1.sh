seed=3407
dataset="CIFAR100"
for model_name in "resnet56" "wrn_40_2" "wrn_16_2" 
do
python3 ./teacher_orig_ds.py --seed=${seed} --model_name=${model_name}\
 --dataset=${dataset}
done 