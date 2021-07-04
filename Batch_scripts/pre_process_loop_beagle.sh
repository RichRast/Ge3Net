#!/bin/bash

# script to pre process the data for all chms

start_chm=$1
end_chm=$2
geno_type=$3

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
        # subset the vcf file with the sample map
        java -Xmx1000g -jar /home/users/richras/packages/beagle.jar gt=$OUT_PATH/${geno_type}/sm_coverage_2/chr${chm}/chr${chm}_sm_coverage_2.vcf.gz out=$OUT_PATH/${geno_type}/sm_coverage_2/chr${chm}/chr${chm}_phased map=$OUT_PATH/${geno_type}/sm_coverage_2/chr${chm}/chr${chm}_sm_coverage_2.map
        # use the ref from previous output and now impute and phase all the samples
        java -Xmx1000g -jar /home/users/richras/packages/beagle.jar ref=$OUT_PATH/${geno_type}/sm_coverage_2/chr${chm}/chr${chm}_phased.vcf.gz out=$OUT_PATH/${geno_type}/chr${chm}/chr${chm}_phased_imputed map=$OUT_PATH/${geno_type}/sm_coverage_2/chr${chm}/chr${chm}_sm_coverage_2.map gt=$OUT_PATH/${geno_type}/chr${chm}/chr${chm}_unfiltered.vcf.gz

    done
fi