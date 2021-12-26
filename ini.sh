#!/bin/bash

# script to set the environment variables for the session
# run this script as source ini.sh

echo "set environment variables"

export USER_PATH="/home/users/richras/Ge2Net_Repo"
export USER_SCRATCH_PATH="/scratch/users/richras"
export IN_PATH='/scratch/groups/cdbustam/richras/data_in'
export OUT_PATH='/scratch/groups/cdbustam/richras/data_out'
export LOG_PATH='/scratch/groups/cdbustam/richras/logs/'
export WANDB_DIR='/scratch/users/richras/Batch_jobs/wandb'
export IMAGE_PATH='/scratch/groups/cdbustam/richras/images'
echo "All done"

# define functions for ad hoc use on the terminal 
json()
{
    # call the function as json params.json
    cat $1 | jq | less
}

csv()
{
    # call the function as csv data.csv
    cat $1 | column -t -s, | less -S

}