#!/bin/bash

# script to create backups every 30 days or on manual trigger
# should be run from Ge2Net_Repo dir structure

source ini.sh

# source_server=$1
# target_server=$2
source_dir=$1
target_dir=$2

# touch recursively all the files to update the modified date
# touch source_dir/
#ToDo:automate for automatic backups
sbatch << EOT
#!/bin/bash
#SBATCH --job-name=cron
#SBATCH --begin=now
#SBATCH --dependency=singleton
#SBATCH --time=24:00:00
#SBATCH --mail-type=FAIL
#SBATCH --output=$OUT_PATH/backup.log

## Insert the command to run below. Here, we're just storing the date in a
## cron.log file
if [[ target_server = "nero-mrivas.compute.stanford.edu" ]]; then
    # use sftp instead of ssh
    sftp -put $source_server:$source_dir $target_server:$target_dir
else 
    rsync --exclude '/data_out/archive/' -avP $source_dir $target_dir
fi
echo "Backup complete"
## Resubmit the job for the next execution
# sbatch $0
EOT