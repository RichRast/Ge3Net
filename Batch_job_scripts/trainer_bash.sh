#!/bin/bash

cd /home/users/richras/Ge2Net_Repo

Help()
{
    echo "This script runs the training for Ge2Net"
    echo
    echo "Syntax: scriptTemplate [-e|m|h]"
    echo "options:"
    echo "-e|--expt_id     Specify the data experiment number to run, example 3 "
    echo "-m|--model       Specify the model type and mjor version, example: D6"
    echo "-h|--help        Print this help"
    echo
}

while [[ $# -gt 0 ]]; do
    case $1 in 
    -e | --expt_id ) shift ; exp=$1 ;;
    -m | --model ) shift ; model_type=$1 ;;
    -h | --help ) Help ; exit ;;
    \? ) echo "Error: Invalid option"; exit 1;;
    esac; shift
done

echo "Checking"

if [[ -z $exp ]] ; then echo "Missing data experiment id for which to run the experiment" ; exit 1; fi
if [[ -z $model_type ]] ; then echo "Missing model type" ; exit 1; fi

echo "Starting experiment with Model $model_type and data from experiment # $exp"

sbatch << EOT
#!/bin/bash
#SBATCH -p gpu
#SBATCH -c 10
#SBATCH -G 1
#SBATCH --mem=80G
#SBATCH -t 24:00:00
#SBATCH --output=slurm-%a_%j_exp_$exp_$model_type.out

ml load py-pytorch/1.4.0_py36
ml load py-scipy/1.4.1_py36
ml load py-matplotlib/3.2.1_py36
ml load py-pandas/1.0.3_py36
ml load cuda/10.1.168
ml load git-lfs/2.4.0

python3 trainer.py --data.experiment_id $exp \
--data.params_dir '$USER_PATH/experiments/pca/exp_$model_type/' \
--data.experiment_name 'unsupervised_labeling' \
--data.labels_dir '$OUT_PATH/unsupervised_labeling/$exp'
EOT

squeue -u richras