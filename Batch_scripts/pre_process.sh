#!/bin/bash

# script to pre-process vcf data so that it is ready to be used to 
# create labels using unsupervised methods
# sample script ./Batch_scripts/pre_process.sh st_chm=1 ed_chm=38 sm=expt1 geno=0.1 maf=0.01 ld_prune=0.5 combine geno_type=dogs

cd /home/users/richras/Ge2Net_Repo
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
        c|combine )             combine="True";;
        sm|sample_map )         sample_map=$value;;
        geno )                  geno=$value;;
        maf )                   maf=$value;;
        ld_prune )              ld_prune=$value;;
        gt|geno_type )          geno_type=$value;;
        \? ) echo "Error: Invalid options"; exit 1;;
    esac
done

echo "Checking"

if [[ -z $start_chm ]] ; then echo "Missing start chm number" ; exit 1; fi
if [[ -z $end_chm ]] ; then echo "Missing end chm numberd" ; exit 1 ; fi
if [[ -z $combine ]] ; then echo "Missing combine argument, no chms will be combined" ; combine="False"; fi
if [[ -z $sample_map ]] ; then echo "Missing sample map to subset" ; exit 1; fi
if [[ -z $geno ]] ; then echo "Missing mind parameter ()" ; exit 1; fi
if [[ -z $maf ]] ; then echo "Missing maf parameter" ; exit 1; fi
if [[ -z $ld_prune ]] ; then echo "Missing ld prune parameter" ; exit 1; fi
if [[ -z $geno_type ]] ; then echo "Missing geno type" ; exit 1; fi

sbatch<<EOT
#!/bin/bash
#SBATCH -p bigmem
#SBATCH -c 1
#SBATCH --mem=1000G
#SBATCH -t 24:00:00
#SBATCH --output=$OUT_PATH/dogs/preprocess_vcf_sm_${sample_map}_${start_chm}_${end_chm}_combine_${combine}_ld_prune_${ld_prune}.out

ml load biology
ml load plink/2.0a2
ml load bcftools/1.8
ml load vcftools/0.1.15

cd /home/users/richras/Ge2Net_Repo

if ./Batch_scripts/pre_process_loop.sh ${start_chm} ${end_chm} ${geno} ${maf} ${ld_prune} ${sample_map} ${geno_type}; then echo "Success";
else echo "Fail"; fi
if [[ ($combine = "True") ]] ; then
    echo "Launching the combine script for pruned data "
    if ./Batch_scripts/filter_combine_vcf.sh -gt ${geno_type} -sm ${sample_map} -f 0.0 -st_chm 1 -ed_chm 38 -s_win 0 -ld ${ld_prune} -c; then echo "Success";
    else echo "Fail combine script"; fi 
    echo "Launching the combine script for unpruned and unfiltered data"
    if ./Batch_scripts/filter_combine_vcf.sh -gt ${geno_type} -sm ${sample_map} -f 0.0 -st_chm 1 -ed_chm 38 -s_win 0 -c ; then echo "Success";
    else echo "Fail combine script"; fi 
else echo "Finished without combining";
fi

EOT
