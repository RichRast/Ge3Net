#!/bin/bash

# script to create backups every 30 days or on manual trigger
# should be run from Ge2Net_Repo dir structure

source ini.sh

echo " cd into dir ${USER_SCRATCH_PATH}"
    select dir2backup in "data_in" "data_out" "both"; do
        case $dir2backup in
            data_in ) echo " backing up data_in folder"
            data_out ) echo " backing up data_out folder"
            both ) echo " backing up both folder"
        esac
    done

#!/bin/bash
#SBATCH --job-name=cron
#SBATCH --begin=now+7days
#SBATCH --dependency=singleton
#SBATCH --time=24:00:00
#SBATCH --mail-type=FAIL


## Insert the command to run below. Here, we're just storing the date in a
## cron.log file
date -R >> $HOME/cron.log

## Resubmit the job for the next execution
sbatch $0