#!/bin/bash
#SBATCH -p gpu
#SBATCH -c 10
#SBATCH -G 1
#SBATCH --mem=80G
#SBATCH -t 24:00:00
#SBATCH --output=slurm-%a_%j_exp_4_D6.out

ml load py-pytorch/1.4.0_py36
ml load py-scipy/1.4.1_py36
ml load py-matplotlib/3.2.1_py36
ml load py-pandas/1.0.3_py36
ml load cuda/10.1.168
ml load git-lfs/2.4.0

cd /home/users/richras/Ge2Net_Repo
python3 trainer.py --train.experiment_id 4 \
--data.params_dir '/home/users/richras/Ge2Net_Repo/experiments/pca/exp_D6/' \
--train.experiment_name 'unsupervised_labeling' \
--data.labels_dir '/scratch/groups/cdbustam/richras/unsupervised_labeling/4'
