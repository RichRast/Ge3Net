#!/bin/bash

# script to pre process the data for all chms

start_chm=$1
end_chm=$2 
geno=$3
maf=$4  
ld_prune=$5
sample_map=$6
geno_type=$7


for (( chm=$1; chm<=$2; chm++ )); do
    echo "processing for chm = ${chm}"
    mkdir -p $OUT_PATH/${geno_type}/sm_${sample_map}/chr${chm}
    # subset the vcf file with the sample map
    cat $OUT_PATH/${geno_type}/ref_map_${sample_map}.tsv | sed '1d' |cut -f 1| bcftools view -S - $IN_PATH/${geno_type}/chr${chm}/chr${chm}_unfiltered_phased.vcf.gz -o $OUT_PATH/${geno_type}/sm_${sample_map}/chr${chm}/chr${chm}_subset_tmp.vcf.gz -O z
    # filter the snps by removing missing snps
    bcftools view -e 'GT[*] = "mis"' -o $OUT_PATH/${geno_type}/sm_${sample_map}/chr${chm}/chr${chm}_biallelic.vcf.gz -O z $OUT_PATH/${geno_type}/sm_${sample_map}/chr${chm}/chr${chm}_subset_tmp.vcf.gz
    echo "Missing snps removed for chm ${chm}" 
    # make the plink directories
    # convert to plink format and remove missing markers as well as samples that miss snps
    # ld pruning
    # convert back to vcf format
    mkdir -p $OUT_PATH/${geno_type}/plink/sm_${sample_map}/chr${chm}
    #clean the plink dir for that chromosome so new data is generated
    rm -v $OUT_PATH/${geno_type}/plink/sm_${sample_map}/chr${chm}/*
    # convert to vcf by making a bed file
    plink2 --vcf $OUT_PATH/${geno_type}/sm_${sample_map}/chr${chm}/chr${chm}_biallelic.vcf.gz --chr-set 38 --set-missing-var-ids @:#[b37]\$r,\$a --make-bed --out $OUT_PATH/${geno_type}/plink/sm_${sample_map}/chr$chm/chr${chm}_biallelic
    plink2 --bfile $OUT_PATH/${geno_type}/plink/sm_${sample_map}/chr$chm/chr${chm}_biallelic --maf $maf --geno $geno --make-bed --out $OUT_PATH/${geno_type}/plink/sm_${sample_map}/chr$chm/chr${chm}_filtered_biallelic --chr-set 38
    # save filtered biallelic as vcf to be used for prediction in downstream such as input for Ge2Net
    plink2 --bfile $OUT_PATH/${geno_type}/plink/sm_${sample_map}/chr$chm/chr${chm}_filtered_biallelic --recode vcf --out $OUT_PATH/${geno_type}/sm_${sample_map}/chr$chm/chr${chm}_filtered --chr-set 38
    plink2 --bfile $OUT_PATH/${geno_type}/plink/sm_${sample_map}/chr$chm/chr${chm}_filtered_biallelic --indep-pairwise 1000 5 ${ld_prune} --out $OUT_PATH/${geno_type}/plink/sm_${sample_map}/chr$chm/data_out_biallelic --chr-set 38
    plink2 --bfile $OUT_PATH/${geno_type}/plink/sm_${sample_map}/chr$chm/chr${chm}_filtered_biallelic --extract $OUT_PATH/${geno_type}/plink/sm_${sample_map}/chr$chm/data_out_biallelic.prune.in --make-bed --out $OUT_PATH/${geno_type}/plink/sm_${sample_map}/chr$chm/pruneddata_biallelic --chr-set 38
    plink2 --bfile $OUT_PATH/${geno_type}/plink/sm_${sample_map}/chr$chm/pruneddata_biallelic --recode vcf --out $OUT_PATH/${geno_type}/sm_${sample_map}/chr$chm/vcf_pruned_chr$chm --chr-set 38

done