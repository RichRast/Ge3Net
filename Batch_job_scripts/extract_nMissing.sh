#!/bin/bash

# This script calculates the stats for each chm of dogs and extracts the sample
# with the nMissing per chm

# sample command: ./Batch_job_scripts/extract_nMissing.sh <start_chm> <end_chm>

cd /home/users/richras/Ge2Net_Repo
source ini.sh

echo "Loading libraries for loading"
ml load biology
ml load bcftools/1.8

for (( chm=$1; chm<=$2; chm++ )); do
    echo "computing biostats for chm $chm"
    # compute the stats and store it in the chrx_biostats.txt
    bcftools stats -s - $OUT_PATH/dogs/chr$chm/chr${chm}_subset.vcf.gz >$OUT_PATH/dogs/chr$chm/chr${chm}_biostats.txt

    echo "Extract number of snps, sample and nMissing for chm $chm"
    # extract number of snps for the chm
    sed -n '26p' $OUT_PATH/dogs/chr$chm/chr${chm}_biostats.txt | cut -f3- >$OUT_PATH/dogs/chr$chm/chr${chm}_nMissing.txt
    # for the particular subset, it is always lines 1718 through 2384 and column 3, 14 
    # for sample, nMissing
    sed -n '1718,2384p' $OUT_PATH/dogs/chr$chm/chr${chm}_biostats.txt | cut -f3,14 >>$OUT_PATH/dogs/chr$chm/chr${chm}_nMissing.txt
done

echo "All Done"

exit 0



