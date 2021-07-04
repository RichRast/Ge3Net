#!/bin/bash

# sample command srun --pty -p bigmem --mem=1100G -c 1 --time 24:00:00 ./Batch_scripts/ancient_preProcess.sh ancient 1 24 sm_coverage_2
# convert .ped to .vcf.gz
# use bcftools to subset by chromosomes
source ini.sh

geno_type=$1
start_chm=$2
end_chm=$3
sample_map=$4

echo "start_chm $start_chm"
echo "end_chm $end_chm"

ml load biology
ml load plink/2.0a2
ml load bcftools/1.8

#form the sample map from tsv
mkdir -p $OUT_PATH/${geno_type}/$sample_map
sed '1d' $OUT_PATH/${geno_type}/${sample_map}.tsv | cut -f 1 > $OUT_PATH/${geno_type}/$sample_map/sample_id.txt

for (( chm=$start_chm; chm<=$end_chm; chm++ )); do
    echo "processing for chm = ${chm}" 
    mkdir -p $OUT_PATH/${geno_type}/$sample_map/chr${chm} 
    # subset the unfiltered vcf by sample map
    bcftools view -o $OUT_PATH/${geno_type}/$sample_map/chr${chm}/chr${chm}_$sample_map.vcf.gz -O z -S $OUT_PATH/${geno_type}/$sample_map/sample_id.txt $OUT_PATH/${geno_type}/chr${chm}/chr${chm}_unfiltered.vcf.gz
    
done