#!/bin/bash

# sample command srun --pty -p gpu --gres=gpu:1 --mem=250GB --time 24:00:00 ./Batch_scripts/trainer_bash_srun.sh -gt humans -e 2 -d 1_geo -m A -sum "model_A" -wandb_name "Ge3Net_v2" -v
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
    echo "-sum|--summary  Specify summary or description of this run"
    echo "-pt_dir|--pretrain_dir  Specify pretrain dir" 
    echo "-wandb_name|--wandb_name  Specify the name of wandb project"
    echo "-v|--verbose     Specify True, False for verbose "
    echo "-h|--help        Print this help"
    echo
}

while [[ $# -gt 0 ]]; do
    case $1 in 
    -gt | --geno_type ) shift ; geno_type=$1 ;;
    -d | --data_id ) shift ; data_id=$1 ;;
    -e | --expt_id ) shift ; expt_id=$1 ;;
    -m | --model ) shift ; model_type=$1 ;;
    -sum | --summary ) shift ; model_summary=$1 ;;
    -pt_dir | --pretrain_dir ) shift; pretrain_dir=$1 ;;
    -wandb_name|--wandb_name ) shift; wandb_name=$1 ;;
    -v | --verbose ) shift ; verbose="True";;
    -h | --help ) Help ; exit ;;
    \? ) echo "Error: Invalid option"; exit 1;;
    esac; shift
done

echo "Checking arguments for running trainer"

if [[ -z $data_id ]] ; then echo "Missing data experiment id for which to run the experiment" ; exit 1; fi
if [[ -z $expt_id ]] ; then echo "Missing experiment id for Ge2Net training" ; exit 1; fi
if [[ -z $model_type ]] ; then echo "Missing model type" ; exit 1; fi
if [[ -z $geno_type ]] ; then echo "Setting default genotype to humans" ; geno_type='humans' ; exit ; fi
if [[ -z $verbose ]] ; then echo "Setting verbose to default of False" ; verbose='False'; fi
if [[ -z $pretrain_dir ]]; then echo "no pretrained dir specified, training from scratch" ; pretrain_dir='None'; fi
if [[ -z $wandb_name ]]; then echo "no wandb_name specified, setting to Ge3Net_v2" ; wandb_name='Ge3Net_v2'; fi

echo "Starting experiment $expt_id with Model $model_type and data from experiment # $data_id for geno_type $geno_type"

if [[ ("$data_id" = *"_geo"*) ]]; then
    params_name="params"; 
elif [[ ("$data_id" = *"_pca"*) ]]; then
    params_name="params_pca";
elif [[ ("$data_id" = *"_umap"*) ]]; then
    params_name="params_umap";
else
    echo "exiting"; exit 1 ;
fi
echo "data_id $data_id params_name $params_name"

if [[ -d $OUT_PATH/${geno_type}/training/Model_${model_type}_exp_id_${expt_id}_data_id_${data_id} ]];
then
    echo " $OUT_PATH/${geno_type}/training/Model_${model_type}_exp_id_${expt_id}_data_id_${data_id} already exists. Are you sure you want to overwrite ?";
    select yn in "Yes" "No"; do
        case $yn in
            Yes ) echo "okay going to overwrite and continue to start training"; break;;
            No ) echo "okay, exiting"; exit;;
        esac
    done
else
    echo "$OUT_PATH/${geno_type}/training/Model_${model_type}_exp_id_${expt_id}_data_id_${data_id} doesn't exist, creating it";
    mkdir -p $OUT_PATH/${geno_type}/training/Model_${model_type}_exp_id_${expt_id}_data_id_${data_id}
    echo "dir created"
fi

ml load py-pytorch/1.4.0_py36
ml load py-scipy/1.4.1_py36
ml load py-matplotlib/3.2.1_py36
ml load py-pandas/1.0.3_py36
ml load cuda/10.1.168
ml load git-lfs/2.4.0
ml load system nvtop

python3 trainer.py  --data.params $USER_PATH/src/main/experiments/exp_$model_type/${params_name}.yaml \
--data.geno_type $geno_type \
--data.labels $OUT_PATH/$geno_type/labels/data_id_${data_id} \
--data.dir $OUT_PATH/$geno_type/labels/data_id_${data_id} \
--models.dir $OUT_PATH/$geno_type/training/Model_${model_type}_exp_id_${expt_id}_data_id_${data_id}/ \
--model.summary $model_summary \
--model.pretrained_dir $pretrain_dir \
--model.wandb_name $wandb_name \
--log.verbose $verbose 2>&1 | tee $OUT_PATH/$geno_type/training/Model_${model_type}_exp_id_${expt_id}_data_id_${data_id}/Ge3Net.log

node_feat -n $(hostname|sed 's/.int.*//') >> $OUT_PATH/$geno_type/training/Model_${model_type}_exp_id_${expt_id}_data_id_${data_id}/Ge3Net.log

