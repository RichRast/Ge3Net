#!/bin/bash
source ini.sh
awk '{print $1,$2,$3,$4}' $IN_PATH/ancient/ped_format/v44.3_1240K_ancient.pedsnp > $IN_PATH/ancient/ped_format/v44.3_1240K_ancient.map