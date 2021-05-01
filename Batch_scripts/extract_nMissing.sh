#!/bin/bash

# This script calculates the stats for each chm of dogs and extracts the sample
# with the nMissing per chm

# sample command: ./Batch_scripts/extract_nMissing.sh <geno_type> <sample_map> <start_chm> <end_chm>
# ./Batch_scripts/extract_nMissing.sh dogs a 1 38

cd /home/users/richras/Ge2Net_Repo
source ini.sh

echo "Loading libraries for loading"
ml load biology
ml load bcftools/1.8

geno_type=$1
sample_map=$2
for (( chm=$3; chm<=$4; chm++ )); do
    echo "computing biostats for chm $chm"
    # compute the stats and store it in the chrx_biostats.txt
    bcftools stats -s - $OUT_PATH/${geno_type}/sm_${sample_map}/chr$chm/chr${chm}_subset_tmp.vcf.gz >$OUT_PATH/${geno_type}/sm_${sample_map}/chr$chm/chr${chm}_biostats.txt

    echo "Extract number of snps, sample and nMissing for chm $chm"
    # extract number of snps for the chm
    sed -n '26p' $OUT_PATH/${geno_type}/sm_${sample_map}/chr$chm/chr${chm}_biostats.txt | cut -f3- >$OUT_PATH/${geno_type}/sm_${sample_map}/chr$chm/chr${chm}_nMissing.txt
    # for the particular subset, it is always lines 1718 through 2384 and column 3, 14 
    # for sample, nMissing
    linenum=$(awk '/nMissing/{ print NR; exit }' $OUT_PATH/${geno_type}/sm_${sample_map}/chr$chm/chr${chm}_nMissing.txt)
    echo "linenum $linenum"
    sed -n '1249,1446p' $OUT_PATH/${geno_type}/sm_${sample_map}/chr$chm/chr${chm}_biostats.txt | cut -f3,14 >>$OUT_PATH/${geno_type}/sm_${sample_map}/chr$chm/chr${chm}_nMissing.txt
done

echo "All Done"

exit 0



