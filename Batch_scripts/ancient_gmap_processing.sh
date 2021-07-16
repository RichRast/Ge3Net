#!/bin/bash
source ini.sh
#sample command ./Batch_scripts/ancient_gmap_processing.sh ancient 1 24 sm_coverage_2_HG_NA

# remove last two columns 5 and 6
# remove chm 24
# awk '{print $1,$2,$3,$4}' $IN_PATH/ancient/ped_format/v44_ancient.pedsnp|awk '$1 != 24' > $IN_PATH/ancient/ped_format/v44_ancient.map
# echo "Finish"
geno_type=$1
start_chm=$2
end_chm=$3
sample_map=$4

for (( chm=${start_chm}; chm<=${end_chm}; chm++ )); do
    echo "processing for chm = ${chm}"
    awk '{print$1,".",$3*100,$4}' $IN_PATH/${geno_type}/ped_format/v44_ancient.pedsnp | awk -v chm=${chm} '$1 == chm'> $OUT_PATH/${geno_type}/${sample_map}/chr${chm}/chr${chm}_${sample_map}.map
    echo "processing for simulation for chm = ${chm}"
    awk '{print$1,$4,$3}' $OUT_PATH/${geno_type}/${sample_map}/chr${chm}/chr${chm}_${sample_map}.map > $OUT_PATH/${geno_type}/chr${chm}/chr${chm}_phased_imputed.map
done

echo "All done"
