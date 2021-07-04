#!/bin/bash
source ini.sh
#sample command ./Batch_scripts/ancient_gmap_processing.sh ancient 1 24

# remove last two columns 5 and 6
# remove chm 24
# awk '{print $1,$2,$3,$4}' $IN_PATH/ancient/ped_format/v44_ancient.pedsnp|awk '$1 != 24' > $IN_PATH/ancient/ped_format/v44_ancient.map
# echo "Finish"
geno_type=$1
start_chm=$2
end_chm=$3

for (( chm=${start_chm}; chm<=${end_chm}; chm++ )); do
    echo "processing for chm = ${chm}"
    awk '{print $1,$2,$3,$4}' $IN_PATH/${geno_type}/ped_format/v44_ancient.pedsnp | awk -v chm=$chm '$1 == chm' > $OUT_PATH/${geno_type}/sm_coverage_2/chr${chm}/chr${chm}_sm_coverage_2.map
done

echo "All done"