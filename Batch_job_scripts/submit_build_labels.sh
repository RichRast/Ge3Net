#!/bin/bash
#SBATCH -p bigmem
#SBATCH -c 1
#SBATCH --mem=1000G
#SBATCH -t 24:00:00
#SBATCH --output=slurm-%a_%j_exp_build_lbls_3.out

ml load py-pytorch/1.4.0_py36
ml load py-scipy/1.4.1_py36
ml load py-numpy/1.18.1_py36
ml load py-matplotlib/3.2.1_py36
ml load py-pandas/1.0.3_py36

cd /home/users/richras/Ge2Net_Repo
python3 build_labels_revised.py --data.experiment_id 3 \
--data.all_chm_snps '/scratch/groups/cdbustam/richras/combined_chm/all_chm_combined_snps_variance_filter_0.09.npy' 
