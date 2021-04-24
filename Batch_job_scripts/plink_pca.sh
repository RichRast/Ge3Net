#!/bin/bash
# sample command: ./Batch_job_scripts/plink_pca.sh <chm> <vcf_type>
# ./Batch_job_scripts/plink_pca.sh 1 expt1 dogs
# vcf_type can be "subset" or "biallelic" or "expt1_filtered" 

cd /home/users/richras/Ge2Net_Repo
source ini.sh
chm=$1
vcf_type=$2
geno_type=$3

sbatch<<EOT
#!/bin/sh
#SBATCH -p bigmem
#SBATCH -c 1
#SBATCH --mem=1000G
#SBATCH -t 24:00:00
#SBATCH --output=$OUT_PATH/dogs/plink_pca_${chm}_${vcf_type}.out

echo "Loading libraries for loading"
ml load biology
ml load plink/2.0a2
ml load bcftools/1.8

plink2 --vcf $OUT_PATH/${geno_type}/${vcf_type}_combined.vcf.gz --chr-set 38 --make-bed --out $OUT_PATH/dogs/plink/${vcf_type}_combined_plink
plink2 --bfile $OUT_PATH/dogs/plink/${vcf_type}_combined_plink --pca 10 --out $OUT_PATH/dogs/plink/plinkpca_${vcf_type} --chr-set 38


EOT

sleep .5
squeue -u richras