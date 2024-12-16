# An Adversarial Model with Diffusion for Robust Recommendation against Shilling Attack
This is the pytorch implementation of our paper at SIGAPP 2025:
> [An Adversarial Model with Diffusion for Robust Recommendation against Shilling Attack](https://doi.org/10.1145/3672608.3707821)
> 
> Thi-Hanh Le, Padipat Sitkrongwong, Panagiotis Andriotis, Quang-Thuy Ha, Atsuhiro Takasu

## Environment
- Anaconda 3
- python 3.8.10
- pytorch 1.12.0
- numpy 1.22.3

## Usage
### Data
The experimental data are in '../datasets' folder, including Amazon-apps.

### Training
To reproduce the results and perform fine-tuning of the hyperparameters, please refer to the model name specified in the **inference.py** file. Ensure that the hyperparameter 'noise_min' is set to a value lower than 'noise_max'.
#### DiffRec
```
cd ./Diff-WGAN
python main.py --cuda --dataset=$1 --data_path=../datasets/$1/ --lr=$2 --lr2=$3 --weight_decay=$4 --batch_size=$5 --dims=$6 --emb_size=$7 --mean_type=$8 --steps=$9 --noise_scale=$10 --noise_min=${11} --noise_max=${12} --sampling_steps=${13} --reweight=${14} --log_name=${15} --round=${16} --gpu=${17}

```
or use run.sh
```
cd ./Diff-WGAN
sh run.sh dataset lr_generator lr_discriminator weight_decay batch_size dims emb_size mean_type steps noise_scale noise_min noise_max sampling_steps reweight log_name round gpu_id
```

### Inference

1. Download the checkpoints released by us from [Google Drive](https://drive.google.com/drive/folders/1zlrid4jmbwGCtQWW1dzHEIRrA6VU3Pue?usp=sharing).
2. Put it in  the 'checkpoints' folder.
3. Run inference.py
```
python inference.py --dataset=$1 --gpu=$2
```

### Examples

1. Train Diff-WGAN on Amazon-apps under clean setting
```
cd ./Diff-WGAN
sh run.sh amazon-apps_clean 5e-5 1e-5 0 1000 [1000] 10 x0 5 0.0001 0.0005 0.005 0 0 log 1 0
```
2. Inference Diff-WGAN on amazon-apps under aush attack setting
```
cd ./L-DiffRec
python inference.py --dataset=amazon-apps_clean_aush_attack --gpu=0
```

## Citation  
If you use our code, please kindly cite:

```
Thi-Hanh Le, Padipat Sitkrongwong, Panagiotis Andriotis, Quang-Thuy Ha, and Atsuhiro Takasu. 2025. An Adversarial Model with Diffusion for Robust Recommendation against Shilling Attack. In The 40th ACM/SIGAPP
Symposium on Applied Computing (SAC ’25), March 31-April 4, 2025, Catania, Italy. ACM, New York, NY, USA, 8 pages. https://doi.org/10.1145/3672608.3707821
```


```
@inproceedings{wang2023diffrec,
title = {Diffusion Recommender Model},
author = {Wang, Wenjie and Xu, Yiyan and Feng, Fuli and Lin, Xinyu and He, Xiangnan and Chua, Tat-Seng},
booktitle = {Proceedings of the 46th International ACM SIGIR Conference on Research and Development in Information Retrieval},
pages = {832–841},
publisher = {ACM},
year = {2023}
}
```
