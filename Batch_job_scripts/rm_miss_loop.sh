#!/bin/bash

for (( chm=$1; chm<=$2; chm++ )); do
    echo chm ${chm}
    mkdir -p $OUT_PATH/dogs/chr${chm}
    # subset the vcf to the particular sample map
    cat $OUT_PATH/dogs/ref_map_keep.txt | cut -f 1| bcftools view -S - $IN_PATH/dogs/chr${chm}/chr${chm}_unfiltered.vcf.gz -o $OUT_PATH/dogs/chr${chm}/chr${chm}_subset.vcf.gz -O z
    # filter the snps by removing missing snps
    bcftools view -e 'GT[*] = "mis"' -o $OUT_PATH/dogs/chr${chm}/chr${chm}_biallelic.vcf.gz -O z $OUT_PATH/dogs/chr${chm}/chr${chm}_subset.vcf.gz
    echo "Missing snps removed for chm ${chm}" 
done