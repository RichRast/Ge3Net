#!/bin/bash

cd /home/users/richras/Ge2Net_Repo

Help()
{
    echo "This script filters the vcf and or combines vcf of multiple chromosomes for Ge2Net"
    echo
    echo "Syntax: scriptTemplate [-gt|f|c|st_chm|ed_chm|h]"
    echo "options:"
    echo "-gt|--geno_type            Specify whether the genotype is 'humans' or 'dogs"
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
    -f|--variance_filter ) shift ; filter=$1 ;;
    -c|--combine  ) shift ; combine="True" ;;
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

if [[ ($combine -eq "False") && ($filter -eq "0") ]] ; then echo "Invalid arguments, exiting" ; exit 1 ; fi

echo "Starting with filter = $filter and combine= $combine"
mkdir -p $OUT_PATH/$geno_type

sbatch<<EOT
#!/bin/sh
#SBATCH -p bigmem
#SBATCH -c 1
#SBATCH --mem=1000G
#SBATCH -t 24:00:00
#SBATCH --output=$OUT_PATH/$geno_type/chm_${filter}_combine_${combine}_chm_${start_chm}_chm_${end_chm}.out

ml load py-pytorch/1.4.0_py36
ml load py-scipy/1.4.1_py36
ml load py-numpy/1.18.1_py36
ml load py-matplotlib/3.2.1_py36
ml load py-pandas/1.0.3_py36

cd /home/users/richras/Ge2Net_Repo

python3 combine_chms.py --data.geno_type $geno_type \
--data.variance_filter $filter \
--data.combine $combine \
--data.chm_start $start_chm \
--data.chm_end $end_chm

EOT

sleep .5
squeue -u richras

exit 0