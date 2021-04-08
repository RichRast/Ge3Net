#!/bin/bash

# script for dogs to extract biallelic snps from unfiltered
# with sample map obtained by removing [ 'Wolf', 'Coyote', 'Dhole', 'Andean Fox', 'Jackal']
# sample command: ./Batch_job_scripts/rm_missing_snp.sh -st_chm=1 -ed_chm=38

cd /home/users/richras/Ge2Net_Repo
source ini.sh

for arg in "$@" ; do
    key=${arg%%=*}
    value=${arg#*=}

    echo "$key"
    echo "$value"
    echo "*******************"

    case "$key" in
        -st_chm|--start_chm )          start_chm=$value;;
        -ed_chm|--end_chm )            end_chm=$value;;        
        \? ) echo "Error: Invalid option"; exit 1;;
    esac    
done

if [[ -z ${start_chm} ]] ; then echo "Missing start chm "; exit 1 ; fi
if [[ -z ${end_chm} ]] ; then echo "Missing end chm "; exit 1 ; fi
echo "start_chm ${start_chm} and end_chm ${end_chm}"

sbatch<<EOT
#!/bin/sh
#SBATCH -p bigmem
#SBATCH -c 1
#SBATCH --mem=1000G
#SBATCH -t 24:00:00
#SBATCH --output=$OUT_PATH/dogs/subset_vcf_${start_chm}_${end_chm}.out

ml load biology
ml load plink/2.0a2
ml load bcftools/1.8

cd /home/users/richras/Ge2Net_Repo

chmod u+x ./Batch_job_scripts/rm_miss_loop.sh
if ./Batch_job_scripts/rm_miss_loop.sh ${start_chm} ${end_chm} ; then echo "Success" ;
else echo "Fail"; fi

EOT
echo "All done"

sleep .5
squeue -u richras

exit 0