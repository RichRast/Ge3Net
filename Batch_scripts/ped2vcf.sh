#!/bin/bash

# sample command srun --pty -p bigmem --mem=1100G -c 1 --time 24:00:00 ./Batch_scripts/ped2vcf.sh ancient 1 24
# convert .ped to .vcf.gz
# use bcftools to subset by chromosomes
source ini.sh

geno_type=$1
start_chm=$2
end_chm=$3

echo "start_chm $start_chm"
echo "end_chm $end_chm"

ml load biology
# ml load plink/1.90b5.3
ml load plink/2.0a2
ml load bcftools/1.8

#convert from ped to bed using plink 1.9
# mkdir -p $OUT_PATH/${geno_type}/plink
# convert ped to bed and recode 1234 to ACGT
# plink --file $IN_PATH/${geno_type}/ped_format/v44_ancient --make-bed --alleleACGT --out $OUT_PATH/${geno_type}/plink/v44_ancient_vcf 
# recode from bed to vcf using plink2
# plink2 --bfile $OUT_PATH/${geno_type}/plink/v44_ancient_vcf  --recode vcf --out $OUT_PATH/${geno_type}/plink/v44_ancient_vcf_recoded 
# # convert from vcf to vcf.gz
# bcftools view $OUT_PATH/${geno_type}/plink/v44_ancient_vcf_recoded.vcf -O z -o $OUT_PATH/${geno_type}/plink/v44_ancient_vcf_recoded.vcf.gz
# # get the tabix index file for vcf.gz
# bcftools index $OUT_PATH/${geno_type}/plink/v44_ancient_vcf_recoded.vcf.gz


for (( chm=$start_chm; chm<=$end_chm; chm++ )); do
    echo "processing for chm = ${chm}" 
    mkdir -p $OUT_PATH/${geno_type}/chr${chm} 
    # get chromsome, only bi-allelic snps and format them with only genotype (GT) information
    echo "Subsetting into chm"
    bcftools view -v snps -m2 -M2 --regions ${chm} $OUT_PATH/${geno_type}/plink/v44_ancient_vcf_recoded.vcf.gz |  bcftools annotate -o $OUT_PATH/${geno_type}/chr${chm}/chr${chm}_unfiltered.vcf.gz -O z -x INFO,^FORMAT/GT 

done