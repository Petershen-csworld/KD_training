batch_size=128
t_model="resnet50"
s_model="resnet18"
t_init="/home/shenhaoyu/github_projects/KD_training/model/resnet50/resnet50_CIFAR10_seed_3407best.pth"
seed=3407
T=4
lambda_kd=0.9
kd_mode="st"
python3 /home/shenhaoyu/github_projects/KD_training/student_syn_single.py --seed=${seed}\
 --t_init=${t_init} --t_model=${t_model} --s_model=${s_model}\
 --T=${T} --lambda_kd=${lambda_kd}
