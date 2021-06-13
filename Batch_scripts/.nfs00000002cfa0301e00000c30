#!/bin/bash

# script to phase vcf data with beagle 
# sample script ./Batch_scripts/pre_process_beagle.sh st_chm=1 ed_chm=38 geno_type=dogs

source ini.sh

for arg in "$@"; do
    key=${arg%%=*}
    value=${arg#*=}

    echo "key ${key}"
    echo "value ${value}"
    echo "***************"

    case "$key" in
        st_chm|start_chm )      start_chm=$value ;;
        ed_chm|end_chm )        end_chm=$value;;
        gt|geno_type )          geno_type=$value;;
        \? ) echo "Error: Invalid options"; exit 1;;
    esac
done

echo "Checking"

if [[ -z $start_chm ]] ; then echo "Missing start chm number" ; exit 1; fi
if [[ -z $end_chm ]] ; then echo "Missing end chm numberd" ; exit 1 ; fi
if [[ -z $geno_type ]] ; then echo "Missing geno type" ; exit 1; fi

sbatch<<EOT
#!/bin/bash
#SBATCH -p bigmem
#SBATCH -c 1
#SBATCH --mem=1100G
#SBATCH -t 24:00:00
#SBATCH --output=$IN_PATH/dogs/beagle_${start_chm}_${end_chm}.out

ml load biology
ml load java/11.0.11

cd $USER_PATH

if ./Batch_scripts/pre_process_loop_beagle.sh ${start_chm} ${end_chm} ${geno_type}; then echo "Success";
else echo "Fail"; fi

EOT
less +F $IN_PATH/dogs/beagle_${start_chm}_${end_chm}.out
