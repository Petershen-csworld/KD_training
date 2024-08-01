nohup python3 ./teacher_orig_ds.py --model_name=resnet18 \
    --dataset=BloodMNIST --gpu=1 > out/output_note1.log 2>&1 &
nohup python3 ./teacher_orig_ds.py --model_name=resnet20 \
    --dataset=BloodMNIST --gpu=2 > out/output_note2.log 2>&1 &
nohup python3 ./teacher_orig_ds.py --model_name=resnet56 \
    --dataset=BloodMNIST --gpu=3 > out/output_note3.log 2>&1 &
nohup python3 ./teacher_orig_ds.py --model_name=wrn_40_2 \
    --dataset=BloodMNIST --gpu=4 > out/output_note4.log 2>&1 &
nohup python3 ./teacher_orig_ds.py --model_name=wrn_16_2 \
    --dataset=BloodMNIST --gpu=5 > out/output_note5.log 2>&1 &