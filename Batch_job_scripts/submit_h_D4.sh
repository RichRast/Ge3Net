#!/bin/bash
#SBATCH -p gpu
#SBATCH -c 10
#SBATCH -G 1
#SBATCH --mem=80G
#SBATCH -t 20:00:00
#SBATCH --output=slurm-%a_exp_h_D4.out

ml load py-pytorch/1.4.0_py36
ml load py-scipy/1.4.1_py36
ml load py-numpy/1.18.1_py36
ml load py-matplotlib/3.2.1_py36
ml load py-pandas/1.0.3_py36
ml load cuda/10.1.168


cd /home/users/richras/Ge2Net_Repo
python3 hyperparams_optimize.py --train.experiment_id 1 \
--data.params_dir '/home/users/richras/Ge2Net_Repo/experiments/pca/exp_h_D4/' \
--train.experiment_name 'unsupervised_labeling' \ 
--data.labels_dir '/scratch/groups/cdbustam/richras/reference_files/pca_labels'
