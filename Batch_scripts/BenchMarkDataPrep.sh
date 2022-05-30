#!/bin/bash
# srun --pty -p bigmem --mem=1100G -c 1 --time 24:00:00 ./Batch_scripts/BenchMarkDataPrep.sh 2>&1 | tee $OUT_PATH/humans/benchmark/benchMarkDataPrep_data_id_4_geo_log.txt
source ini.sh

ml load biology
ml load bcftools/1.8

StringVal="train_valid test"
dataset="data_id_4_geo"

for split in $StringVal; do
    dir=$OUT_PATH/humans/benchmark/$dataset/$split
    echo "making dir $dir"
    mkdir -p $dir
done

#combine train and valid sample_map.tsv
cat <(tail -n +2 $OUT_PATH/humans/labels/$dataset/train_sample_map.tsv) <(tail -n +2 $OUT_PATH/humans/labels/$dataset/valid_sample_map.tsv) | cut -f 1,2 >$OUT_PATH/humans/benchmark/$dataset/train_valid/sample_map.txt

#remove header from test sample map tsv
sed '1d' $OUT_PATH/humans/labels/$dataset/test_sample_map.tsv|cut -f 1 > $OUT_PATH/humans/benchmark/$dataset/test/sample_map.txt

# copy test admix npy files for gen 0,2,4 and 8
cp -r $OUT_PATH/humans/labels/$dataset/test/* $OUT_PATH/humans/benchmark/$dataset/test/

# use bcftools to subset the vcf file with sample maps for train val and test for the dataset
for split in $StringVal; do
    echo "Subsetting founders for $split"
    bcftools view -o $OUT_PATH/humans/benchmark/$dataset/${split}/founders.vcf.gz -O z -S $OUT_PATH/humans/benchmark/$dataset/$split/sample_map.txt $IN_PATH/humans/master_vcf_files/ref_final_beagle_phased_1kg_hgdp_sgdp_chr22.vcf.gz
done

echo "Finish"