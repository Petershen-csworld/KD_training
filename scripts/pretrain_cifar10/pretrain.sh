seed=3407
dataset="CIFAR10"
for model_name in "resnet18" "resnet20" 
do
python3 ./teacher_orig_ds.py --seed=${seed} --model_name=${model_name}\
 --dataset=${dataset}
done 