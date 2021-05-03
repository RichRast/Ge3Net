#!/bin/bash
# sample command: ./Batch_scripts/plink_pca.sh <chm> <sample_map> <geno_type>
# ./Batch_scripts/plink_pca.sh whole_genome expt1 dogs
# sample_map can be "expt1" or "a" or "b" or "c" or "keep"

cd /home/users/richras/Ge2Net_Repo
source ini.sh
chm=$1
sample_map=$2
geno_type=$3

sbatch<<EOT
#!/bin/sh
#SBATCH -p bigmem
#SBATCH -c 1
#SBATCH --mem=1000G
#SBATCH -t 24:00:00
#SBATCH --output=$OUT_PATH/${geno_type}/plink_pca_${chm}_${sample_map}.out

echo "Loading libraries for loading"
ml load biology
ml load plink/2.0a2
ml load bcftools/1.8

plink2 --vcf $OUT_PATH/${geno_type}/sm_${sample_map}/ld_0.5/combined.vcf.gz --chr-set 38 --make-bed --out $OUT_PATH/${geno_type}/sm_${sample_map}/ld_0.5/combined_plink
plink2 --bfile $OUT_PATH/${geno_type}/sm_${sample_map}/ld_0.5/combined_plink --pca 10 --out $OUT_PATH/${geno_type}/plink/sm_${sample_map}/plinkpca --chr-set 38

EOT

sleep .5
squeue -u richras