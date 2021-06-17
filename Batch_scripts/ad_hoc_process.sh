#!/bin/bash

# script to pre process the data for all chms

source ini.sh

start_chm=$1
end_chm=$2
geno_type=$3

for (( chm=$1; chm<=$2; chm++ )); do
    echo "processing for chm = ${chm}"
    #copy
    # mkdir -p $USER_SCRATCH_PATH/phased_vcf/${geno_type}/chr$chm
    # echo "copying .csi for chm $chm"
    # cp $IN_PATH/${geno_type}/chr${chm}/chr${chm}_unfiltered.vcf.gz.csi $USER_SCRATCH_PATH/phased_vcf/${geno_type}/chr$chm/chr${chm}_unfiltered_phased.vcf.gz.csi
    # echo "copying .vcf for chm $chm"
    # cp $IN_PATH/${geno_type}/chr${chm}/phased.vcf.gz $USER_SCRATCH_PATH/phased_vcf/${geno_type}/chr$chm/chr${chm}_unfiltered_phased.vcf.gz

    #commands to correct the inplace files
    echo "copying the index file for chm$chm"
    cp $IN_PATH/${geno_type}/chr${chm}/chr${chm}_unfiltered.vcf.gz.csi $IN_PATH/${geno_type}/chr${chm}/chr${chm}_unfiltered_phased.vcf.gz.csi 
    
    #rename
    echo "renaming the vcf file for chm$chm"
    mv $IN_PATH/${geno_type}/chr${chm}/phased.vcf.gz $IN_PATH/${geno_type}/chr${chm}/chr${chm}_unfiltered_phased.vcf.gz

done
