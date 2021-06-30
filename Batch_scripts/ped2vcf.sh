#!/bin/bash

# sample command srun --pty -p bigmem --mem=1100G -c 1 --time 24:00:00 ./Batch_scripts/ped2vcf.sh ancient
# convert .ped to .vcf.gz
# use bcftools to subset by chromosomes
source ini.sh
geno_type=$1
ml load biology
ml load plink/1.90b5.3
plink --file $IN_PATH/${geno_type}/ped_format/v44_ancient --recode vcf --out $OUT_PATH/${geno_type}/plink/v44_ancient_vcf