#!/bin/bash

# script to pre process the data for all chms

start_chm=$1
end_chm=$2
geno_type=$3
sample_map=$4

if [[ ${geno_type} = "dogs" ]]; then 
    for (( chm=$1; chm<=$2; chm++ )); do
        echo "processing for chm = ${chm}"
        mkdir -p $OUT_PATH/${geno_type}/sm_${sample_map}/chr${chm}
        # subset the vcf file with the sample map
        java -Xmx1000g -jar /home/users/richras/packages/beagle.jar gt=$IN_PATH/${geno_type}/chr${chm}/chr${chm}_unfiltered.vcf.gz out=$IN_PATH/${geno_type}/chr${chm}/phased map=$IN_PATH/${geno_type}/chr${chm}_beagle_gmap.txt
        
    done
elif [[ ${geno_type} = "ancient" ]]; then 
    for (( chm=$1; chm<=$2; chm++ )); do
        echo "phasing for ${geno_type}"
        mkdir -p $OUT_PATH/${geno_type}/chr${chm}
        # first phase the samples to form the ref panel against which imputation will be performed
        java -Xmx1000g -jar /home/users/richras/packages/beagle.jar gt=$OUT_PATH/${geno_type}/${sample_map}/chr${chm}/chr${chm}_${sample_map}.vcf.gz out=$OUT_PATH/${geno_type}/${sample_map}/chr${chm}/chr${chm}_phased map=$OUT_PATH/${geno_type}/${sample_map}/chr${chm}/chr${chm}_${sample_map}.map
        # use the ref from previous output and now impute and phase all the samples
        java -Xmx1000g -jar /home/users/richras/packages/beagle.jar ref=$OUT_PATH/${geno_type}/${sample_map}/chr${chm}/chr${chm}_phased.vcf.gz out=$OUT_PATH/${geno_type}/chr${chm}/chr${chm}_phased_imputed map=$OUT_PATH/${geno_type}/${sample_map}/chr${chm}/chr${chm}_${sample_map}.map gt=$OUT_PATH/${geno_type}/chr${chm}/chr${chm}_unfiltered.vcf.gz

    done
fi