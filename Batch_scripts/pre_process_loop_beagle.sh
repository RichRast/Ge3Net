#!/bin/bash

# script to pre process the data for all chms

start_chm=$1
end_chm=$2
geno_type=$3

for (( chm=$1; chm<=$2; chm++ )); do
    echo "processing for chm = ${chm}"
    mkdir -p $OUT_PATH/${geno_type}/sm_${sample_map}/chr${chm}
    # subset the vcf file with the sample map
    java -Xmx1000g -jar /home/users/richras/packages/beagle.jar gt=$IN_PATH/dogs/chr${chm}/chr${chm}_unfiltered.vcf.gz out=$IN_PATH/dogs/chr${chm}/phased map=$IN_PATH/dogs/chr${chm}_beagle_gmap.txt
    
done