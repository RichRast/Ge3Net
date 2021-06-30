#!/bin/bash
source ini.sh
# remove last two columns 5 and 6
# remove chm 24
awk '{print $1,$2,$3,$4}' $IN_PATH/ancient/ped_format/v44_ancient.pedsnp|awk '$1 != 24' > $IN_PATH/ancient/ped_format/v44_ancient.map
echo "Finish"
 