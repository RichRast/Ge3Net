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
    mkdir -p $OUT_PATH/${geno_type}/chr${chm}
    # subset the vcf file with the sample map
    cat $OUT_PATH/${geno_type}/ref_map_${sample_map}.txt | cut -f 1| bcftools view -S - $IN_PATH/${geno_type}/chr${chm}/chr${chm}_unfiltered.vcf.gz -o $OUT_PATH/${geno_type}/chr${chm}/chr${chm}_subset_${sample_map}_tmp.vcf.gz -O z
    # filter the snps by removing missing snps
    bcftools view -e 'GT[*] = "mis"' -o $OUT_PATH/${geno_type}/chr${chm}/chr${chm}_biallelic_${sample_map}.vcf.gz -O z $OUT_PATH/${geno_type}/chr${chm}/chr${chm}_subset_${sample_map}_tmp.vcf.gz
    echo "Missing snps removed for chm ${chm}" 
    # make the plink directories
    # convert to plink format and remove missing markers as well as samples that miss snps
    # ld pruning
    # convert back to vcf format
    mkdir -p $OUT_PATH/${geno_type}/plink/chr${chm}
    plink2 --vcf $OUT_PATH/${geno_type}/chr${chm}/chr${chm}_biallelic_${sample_map}.vcf.gz --chr-set 38 --set-missing-var-ids @:#[b37]\$r,\$a --make-bed --out $OUT_PATH/${geno_type}/plink/chr$chm/ex_biallelic
    plink2 --bfile $OUT_PATH/${geno_type}/plink/chr$chm/ex_biallelic --maf $maf --geno $geno --make-bed --out $OUT_PATH/${geno_type}/plink/chr$chm/ex_filtered_biallelic --chr-set 38
    # save filtered biallelic as vcf to be used for prediction in downstream such as input for Ge2Net
    plink2 --bfile $OUT_PATH/${geno_type}/plink/chr$chm/ex_filtered_biallelic --recode vcf --out $OUT_PATH/${geno_type}/chr$chm/chr${chm}_filtered_${sample_map} --chr-set 38
    plink2 --bfile $OUT_PATH/${geno_type}/plink/chr$chm/ex_filtered_biallelic --indep-pairwise 1000 5 ${ld_prune} --out $OUT_PATH/${geno_type}/plink/chr$chm/data_out_biallelic --chr-set 38
    plink2 --bfile $OUT_PATH/${geno_type}/plink/chr$chm/ex_filtered_biallelic --extract $OUT_PATH/${geno_type}/plink/chr$chm/data_out_biallelic.prune.in --make-bed --out $OUT_PATH/${geno_type}/plink/chr$chm/pruneddata_biallelic --chr-set 38
    plink2 --bfile $OUT_PATH/${geno_type}/plink/chr$chm/pruneddata_biallelic --recode vcf --out $OUT_PATH/${geno_type}/chr$chm/vcf_${sample_map}_pruned_chr$chm --chr-set 38

done