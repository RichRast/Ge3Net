#!/bin/bash

# sample command ./Batch_scripts/trainer_bash.sh -gt dogs -e 7 -d 12_test -m D7_dogs_chr22
cd /home/users/richras/Ge2Net_Repo
source ini.sh

Help()
{
    echo "This script runs the training for Ge2Net"
    echo
    echo "Syntax: scriptTemplate [-gt|e|d|m|h]"
    echo "options:"
    echo "-gt|--geno_type  Specify whether the genotype is 'humans' or 'dogs"
    echo "-e|--expt_id     Specify the experiment number for Ge2Net training to run, example 3 "
    echo "-d|--data_id     Specify the data experiment number to run, example 3 "
    echo "-m|--model       Specify the model type and mjor version, example: D6"
    echo "-h|--help        Print this help"
    echo
}

while [[ $# -gt 0 ]]; do
    case $1 in 
    -gt | --geno_type ) shift ; geno_type=$1 ;;
    -d | --data_id ) shift ; data_id=$1 ;;
    -e | --expt_id ) shift ; expt_id=$1 ;;
    -m | --model ) shift ; model_type=$1 ;;
    -h | --help ) Help ; exit ;;
    \? ) echo "Error: Invalid option"; exit 1;;
    esac; shift
done

echo "Checking arguments for running trainer"

if [[ -z $data_id ]] ; then echo "Missing data experiment id for which to run the experiment" ; exit 1; fi
if [[ -z $expt_id ]] ; then echo "Missing experiment id for Ge2Net training" ; exit 1; fi
if [[ -z $model_type ]] ; then echo "Missing model type" ; exit 1; fi
if [[ -z $geno_type ]] ; then echo "Setting default genotype to humans" ; geno_type='humans' ; exit ; fi

echo "Starting experiment $expt_id with Model $model_type and data from experiment # $data_id for geno_type $geno_type"

sbatch << EOT
#!/bin/bash
#SBATCH -p gpu
#SBATCH -c 10
#SBATCH -G 1
#SBATCH -C GPU_MEM:11GB
#SBATCH --mem=80G
#SBATCH -t 24:00:00
#SBATCH --output=$OUT_PATH/logs/gt_${geno_type}_exp_${expt_id}_${model_type}_data_id_${data_id}.out

ml load py-pytorch/1.4.0_py36
ml load py-scipy/1.4.1_py36
ml load py-matplotlib/3.2.1_py36
ml load py-pandas/1.0.3_py36
ml load cuda/10.1.168
ml load git-lfs/2.4.0
ml load system nvtop

# copy yaml params to the path where logs and model are stored

python3 trainer.py --data.experiment_id $data_id \
--data.params_dir '$USER_PATH/experiments/coordinates/exp_$model_type/' \
--data.experiment_name 'unsupervised_labels' \
--data.geno_type $geno_type \
--model.expt_id $expt_id \
--model.working_dir '$OUT_PATH/$geno_type/models_dir' \
--data.labels_dir '$OUT_PATH/$geno_type/sm_expt1/unsupervised_labels/$data_id'
EOT

sleep .5
echo "status of all jobs"
squeue -u richras
sleep .5
echo "status of this job"

echo log_dir: $OUT_PATH/logs/gt_${geno_type}_exp_${expt_id}_${model_type}_data_id_${data_id}.out
less +F $OUT_PATH/logs/gt_${geno_type}_exp_${expt_id}_${model_type}_data_id_${data_id}.out