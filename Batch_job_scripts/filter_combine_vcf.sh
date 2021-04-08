#!/bin/bash

# sample command: ./Batch_job_scripts/filter_combine_vcf.sh -gt dogs -vt expt2_biallelic -f 0.0 -st_chm 1 -ed_chm 38 -c
# or ./Batch_job_scripts/filter_combine_vcf.sh -gt dogs -vt expt1_filtered -f 0.0 -st_chm 1 -ed_chm 38 -c
# vcf_type for dogs can be expt1_filtered, expt2_subset, expt2_biallelic, expt2_subset_pruned, expt2_biallelic_pruned, unfiltered
cd /home/users/richras/Ge2Net_Repo
source ini.sh

Help()
{
    echo "This script filters the vcf and or combines vcf of multiple chromosomes for Ge2Net"
    echo
    echo "Syntax: scriptTemplate [-gt|vt|f|c|st_chm|ed_chm|h]"
    echo "options:"
    echo "-gt|--geno_type            Specify whether the genotype is 'humans' or 'dogs"
    echo "-vt|--vcf_type             Specify the name/type of vcf file used, ex- for dogs - expt1_filtered vs unfiltered"
    echo "-f|--variance_filter       Specify thethreshold for variance above which the snps are discarded"
    echo "-c|--combine               Specify whether the vcf of individual chms need to be combined"
    echo "-st_chm|--start_chm        Specify the start of chm , example 1"
    echo "-ed_chm|--end_chm          Specify the end of chm , example 22"
    echo "-h|--help                  Print this help"
    echo
}

while [[ $# -gt 0 ]]; do
    case $1 in 
    -gt|--geno_type ) shift ; geno_type=$1 ;;
    -vt|--vcf_type ) shift ; vcf_type=$1 ;;
    -f|--variance_filter ) shift ; filter=$1 ;;
    -c|--combine  ) shift 1 ; combine="True" ;;
    -st_chm|--start_chm ) shift ; start_chm=$1 ;;
    -ed_chm|--end_chm ) shift ; end_chm=$1 ;;
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
if [[ (-z $vcf_type) && ($geno_type = "dogs") ]] ; then echo "Missing the vcf type" ; exit 1; fi
if [[ ($combine == "False") && ($filter == 0.0) ]] ; then echo "Invalid arguments, exiting" ; exit 1 ; fi

# form the list of vcf filenames for the st and end chm
# ToDo make this loop less verbose by re-running the missing snp and 
# prune scripts and naming the vcf files accordingly

vcf_filename=()
for chm in $(seq ${start_chm} ${end_chm})
    do
        if [[ ${geno_type} = "humans" ]] ; then
        vcf_filename+=($IN_PATH/${geno_type}/master_vcf_files/ref_final_beagle_phased_1kg_hgdp_sgdp_chr$chm.vcf.gz)
        elif [[ ($geno_type = "dogs") && (${vcf_type} = "expt2_subset") ]] ; then
        vcf_filename+=($OUT_PATH/${geno_type}/chr${chm}/chr${chm}_subset.vcf.gz)
        elif [[ ($geno_type = "dogs") && (${vcf_type} = "expt2_biallelic") ]] ; then
        vcf_filename+=($OUT_PATH/${geno_type}/chr${chm}/chr${chm}_biallelic.vcf.gz)
        elif [[ ($geno_type = "dogs") && (${vcf_type} = "expt2_subset_pruned") ]] ; then
        vcf_filename+=($OUT_PATH/${geno_type}/plink/chr${chm}/vcf_subset_pruned_chr${chm}.vcf)
        elif [[ ($geno_type = "dogs") && (${vcf_type} = "expt2_biallelic_pruned") ]] ; then
        vcf_filename+=($OUT_PATH/${geno_type}/plink/chr${chm}/vcf_biallelic_pruned_chr${chm}.vcf)
        elif [[ ($geno_type = "dogs") && (${vcf_type} = "unfiltered") ]] ; then
        vcf_filename+=($IN_PATH/${geno_type}/chr${chm}/chr${chm}_unfiltered.vcf.gz)
        elif [[ ($geno_type = "dogs") && (${vcf_type} = "expt1_filtered") ]] ; then
        vcf_filename+=($IN_PATH/${geno_type}/chr${chm}/chr${chm}_expt1_filtered.vcf.gz)
        fi
    done
echo ${vcf_filename[*]}

save_path=$OUT_PATH/${geno_type}/${vcf_type}
echo "save path: ${save_path}"
mkdir -p ${save_path}

echo "Starting with filter = $filter and combine= $combine"

sbatch<<EOT
#!/bin/sh
#SBATCH -p bigmem
#SBATCH -c 1
#SBATCH --mem=500G
#SBATCH -t 24:00:00
#SBATCH --output=$OUT_PATH/$geno_type/vcf_${vcf_type}_chm_${filter}_combine_${combine}_chm_${start_chm}_chm_${end_chm}.out

ml load py-pytorch/1.4.0_py36
ml load py-scipy/1.4.1_py36
ml load py-numpy/1.18.1_py36
ml load py-matplotlib/3.2.1_py36
ml load py-pandas/1.0.3_py36

cd /home/users/richras/Ge2Net_Repo

python3 combine_chms.py --data.variance_filter $filter \
--data.vcf_filenames ${vcf_filename[*]} \
--data.save_path ${save_path} \
--data.combine $combine

EOT

sleep .5
squeue -u richras

echo log_dir:$OUT_PATH/$geno_type/vcf_${vcf_type}_chm_${filter}_combine_${combine}_chm_${start_chm}_chm_${end_chm}.out
exit 0