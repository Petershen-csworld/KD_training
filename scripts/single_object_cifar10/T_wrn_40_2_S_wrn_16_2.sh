batch_size=128
t_model="wrn_40_2"
s_model="wrn_16_2"
t_init="/home/shenhaoyu/github_projects/KD_training/model/wrn_40_2/wrn_40_2_CIFAR10_seed_3407best.pth"
seed=3407
T=4
lambda_kd=0.9
kd_mode="st"
python3 /home/shenhaoyu/github_projects/KD_training/student_syn_single.py --seed=${seed}\
 --t_init=${t_init} --t_model=${t_model} --s_model=${s_model}\
 --T=${T} --lambda_kd=${lambda_kd}
