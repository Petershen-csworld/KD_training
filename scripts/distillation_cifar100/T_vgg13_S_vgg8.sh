batch_size=128
dataset="CIFAR10"
t_model="vgg13"
s_model="vgg8"
t_init="/home/shenhaoyu/dataset/model_zoo/pretrained_teacher/vgg13/vgg13_CIFAR100_seed_3407best.pth"
seed=3407
T=4
lambda_kd=0.9
kd_mode="st"
python3 ./student_orig_ds.py --seed=${seed}\
 --t_init=${t_init} --t_model=${t_model} --s_model=${s_model}\
 --T=${T} --lambda_kd=${lambda_kd} --dataset=${dataset}
