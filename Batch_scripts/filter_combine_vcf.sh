#!/bin/bash

# sample command: ./Batch_scripts/filter_combine_vcf.sh -gt dogs -sm expt1 -f 0.0 -st_chm 1 -ed_chm 38 -s_win 100 -c 
# or ./Batch_scripts/filter_combine_vcf.sh -gt dogs -sm expt1 -f 0.0 -st_chm 1 -ed_chm 38 -ld 0.5 -c
# ./Batch_scripts/filter_combine_vcf.sh -gt humans -st_chm 1 -ed_chm 22 -vt ukb -c
# sample_map for dogs can be expt1, a, b, c, d ,e
source ini.sh

Help()
{
    echo "This script filters the vcf and or combines vcf of multiple chromosomes for Ge2Net"
    echo
    echo "Syntax: scriptTemplate [-gt|vt|f|c|s_win|st_chm|ed_chm|h]"
    echo "options:"
    echo "-gt|--geno_type            Specify whether the genotype is 'humans' or 'dogs"
    echo "-sm|--sample_map           Specify the name of sample_map used, ex- for dogs - expt1 or a,b,c"
    echo "-f|--variance_filter       Specify thethreshold for variance above which the snps are discarded"
    echo "-c|--combine               Specify whether the vcf of individual chms need to be combined"
    echo "-s_win|--sample_win        Specify the subsample window size for removing neighbouring correlated snps"
    echo "-st_chm|--start_chm        Specify the start of chm , example 1"
    echo "-ed_chm|--end_chm          Specify the end of chm , example 22"
    echo "-h|--help                  Print this help"
    echo
}

while [[ $# -gt 0 ]]; do
    case $1 in 
    -gt|--geno_type ) shift ; geno_type=$1 ;;
    -sm|--sample_map ) shift ; sample_map=$1 ;;
    -f|--variance_filter ) shift ; filter=$1 ;;
    -c|--combine  ) shift 1 ; combine="True" ;;
    -s_win|--sample_win ) shift 1; sample_win=$1 ;;
    -st_chm|--start_chm ) shift ; start_chm=$1 ;;
    -ed_chm|--end_chm ) shift ; end_chm=$1 ;;
    -ld|--ld_prune ) shift; ld_prune=$1 ;;
    -bcf_combine )  shift; bcf_combine=$1 ;;
    -vt|--vcf_type )  shift; vcf_type=$1 ;;
    -h | --help ) Help ; exit ;;
    \? ) echo "Error: Invalid option"; exit 1;;
    esac; shift
done

echo "Checking"

if [[ -z $geno_type ]] ; then echo "Missing genotype (humans or dogs) for which to run the experiment" ; exit 1; fi
if [[ -z $filter ]] ; then echo "Missing variance filter, no filtering will be performed" ; filter=0.0 ; fi
if [[ -z $combine ]] ; then echo "Missing combine argument, no chms will be combined" ; combine="False"; fi
if [[ -z $start_chm ]] ; then echo "Missing start chm number" ; exit 1; fi
if [[ -z $end_chm ]] ; then echo "Missing end chm number" ; exit 1; fi
if [[ (-z $sample_map) && ($geno_type = "dogs") ]] ; then echo "Missing the vcf type" ; exit 1; fi
if [[ (-z $sample_map) && ($geno_type = "humans") ]] ; then sample_map="None"; fi
if [[ ($combine == "False") && ($filter == 0.0) ]] ; then echo "Invalid arguments, exiting" ; exit 1 ; fi
if [[ -z $sample_win ]] ; then echo "subsample option not selected, setting sample_win to default of 0 "; sample_win=0 ; fi
if [[ -z $ld_prune ]] ; then echo "ld pruning not selected"; ld_prune="False" ; fi
if [[ -z $bcf_combine ]] ; then echo "only combining with python and not using bcftools"; bcf_combine="False"; fi
if [[ -z $vcf_type ]] ; then echo "no specific vcf type specified"; fi

# form the list of vcf filenames for the st and end chm
# ToDo make this loop less verbose by re-running the missing snp and 
# prune scripts and naming the vcf files accordingly

vcf_filename=()
for chm in $(seq ${start_chm} ${end_chm})
    do
        if [[ (${geno_type} = "humans") && (${vcf_type} = "ukb") ]] ; then
        vcf_filename+=($IN_PATH/${geno_type}/ukb/filtered_references/ukb_snps_chm_$chm.recode.vcf)
        elif [[ (${geno_type} = "humans")  ]] ; then
        vcf_filename+=($IN_PATH/${geno_type}/master_vcf_files/ref_final_beagle_phased_1kg_hgdp_sgdp_chr$chm.vcf.gz)
        elif [[ ($geno_type = "dogs") && (${ld_prune} = "False") ]] ; then
        vcf_filename+=($OUT_PATH/${geno_type}/sm_${sample_map}/chr$chm/chr${chm}_filtered.vcf)
        elif [[ ($geno_type = "dogs")]] ; then
        vcf_filename+=($OUT_PATH/${geno_type}/sm_${sample_map}/chr$chm/vcf_pruned_chr$chm.vcf)
        elif [[ ($geno_type = "ancient")]] ; then
        vcf_filename+=($OUT_PATH/${geno_type}/chr$chm/chr${chm}_phased_imputed.vcf.gz)
        fi
    done
echo ${vcf_filename[*]}

save_path=$OUT_PATH/${geno_type}/sm_${sample_map}/ld_${ld_prune}/vcf_type_${vcf_type}
echo "save path: ${save_path}"
mkdir -p ${save_path}

echo "Starting with filter = $filter and combine= $combine"

sbatch<<EOT
#!/bin/sh
#SBATCH -p bigmem
#SBATCH -c 1
#SBATCH --mem=1100G
#SBATCH -t 24:00:00
#SBATCH --output=${save_path}/log.out

ml load py-pytorch/1.4.0_py36
ml load py-scipy/1.4.1_py36
ml load py-numpy/1.18.1_py36
ml load py-matplotlib/3.2.1_py36
ml load py-pandas/1.0.3_py36
ml load biology
ml load bcftools/1.8

cd $USER_PATH

# also combine into a vcf file using bcftools
# this is to inspect the combined vcf's in case for example with dogs
if [[ ${bcf_combine} = True ]] ; then
    echo "bcftools concatenating"
    bcftools concat ${vcf_filename[*]} -O z -o ${save_path}/combined.vcf.gz
fi

echo "Launching python script to combine"
python3 combineChms.py --data.variance_filter $filter \
--data.vcf_filenames ${vcf_filename[*]} \
--data.save_path ${save_path} \
--data.combine $combine \
--data.sample_win $sample_win

EOT

sleep .5
squeue -u richras

echo log_dir:${save_path}/log.out
exit 0