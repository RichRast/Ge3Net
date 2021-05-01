#!/bin/bash

# This script gives the name of each script running on sbatch and 
# if given the option to watch the output file, will open up the output file 
# for that job_id

# sample script ./Batch_scripts/monitor.sh j_id=21823514 -w
username='richras'

for arg in "$@" ; do
    key=${arg%%=*}
    value=${arg#*=}

    echo "key: $key"
    echo "value: $value"
    echo "*******************"

    case "$key" in
        j_id|job_id )          job_id=$value;;
        -w|--watch )           watch="True";;        
        \? ) echo "Error: Invalid option"; exit 1;;
    esac    
done
squeue -u $username

status=$(squeue -u $username | awk -v var="${job_id}" '$0~var{print $5}')
if [[ -z $status ]] ; then
echo "Invalid job id ${job_id}";
else 
echo "job is in $status state";
log_dir=$(scontrol show job ${job_id} | awk -F = '/StdOut=/{print $2}')
echo "log dir:$log_dir "
fi

if [[ ${watch} = "True" && ${status} = "R" ]]; then
less +F $log_dir ;
fi