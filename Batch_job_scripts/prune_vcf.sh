#!/bin/bash

# sample command: ./prune_vcf.sh <st_chm> <ed_chm> <vcf_type>
# vcf_type can be "subset" or "biallelic"

cd /home/users/richras/Ge2Net_Repo
source ini.sh

sbatch<<EOT
#!/bin/sh
#SBATCH -p bigmem
#SBATCH -c 1
#SBATCH --mem=1000G
#SBATCH -t 24:00:00
#SBATCH --output=$OUT_PATH/dogs/vcf_$3_prune_$1_$2.out

echo "Loading libraries for loading"
ml load biology
ml load plink/2.0a2
ml load bcftools/1.8

cd /home/users/richras/Ge2Net_Repo

chmod u+x prune_vcf_loop.sh

if ./prune_vcf_loop.sh $1 $2 $3 ; then echo "Success" ;
else echo "Fail"; fi

EOT

sleep .5
squeue -u richras