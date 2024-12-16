nohup python -u main.py --cuda --dataset=$1 --data_path=../datasets/$1/ --lr=$2 --lr2=$3 --weight_decay=$4 --batch_size=$5 --dims=$6 --emb_size=$7 --mean_type=$8 --steps=$9 --noise_scale=$10 --noise_min=${11} --noise_max=${12} --sampling_steps=${13} --reweight=${14} --log_name=${15} --round=${16} --gpu=${17} > log/$1/gan_${16}_$1_lr$2_lr2$3_wd$4_bs$5_dims$6_emb$7_$8_steps$9_scale$10_min${11}_max${12}_sample${13}_reweight${14}_${15}.txt 2>&1 &

# sh run.sh amazon-apps_clean 5e-5 1e-5 0 1000 [1000] 10 x0 5 0.0001 0.0005 0.005 0 0 log 1 6

