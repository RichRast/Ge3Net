#!/bin/bash

# while [[ $chm -le $2 ]]
for (( chm=$1; chm<=$2; chm++ ));
do 
    # convert vcf to plink format
    # prune plink vcf
    # convert pruned file to an output file
    # convert plink back to vcf
    echo "Pruning snps for chm $chm "
    mkdir -p $OUT_PATH/dogs/plink/chr$chm
    plink2 --vcf $OUT_PATH/dogs/chr${chm}/chr${chm}_$3.vcf.gz --chr-set 38 --make-bed --out $OUT_PATH/dogs/plink/chr$chm/ex_$3
    plink2 --bfile $OUT_PATH/dogs/plink/chr$chm/ex_$3 --indep-pairwise 50 5 0.2 --out $OUT_PATH/dogs/plink/chr$chm/data_out_$3 --chr-set 38
    plink2 --bfile $OUT_PATH/dogs/plink/chr$chm/ex_$3 --extract $OUT_PATH/dogs/plink/chr$chm/data_out_$3.prune.in --make-bed --out $OUT_PATH/dogs/plink/chr$chm/pruneddata_$3 --chr-set 38
    plink2 --bfile $OUT_PATH/dogs/plink/chr$chm/pruneddata_$3 --recode vcf --out $OUT_PATH/dogs/plink/chr$chm/vcf_$3_pruned_chr$chm --chr-set 38
    
done
echo "All done"