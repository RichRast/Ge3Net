#!/bin/bash
source ini.sh

# sample command ./Batch_scripts/build_labels.sh -gt=dogs -e=1 -sim -bl -n_o=3 -sm=expt1 -s=1234 -um=umap -st_chm=22 -ed_chm=22
#./Batch_scripts/build_labels.sh -gt=humans -e=1 -sim -bl -n_o=3 -um=pca -vt=ukb -st_chm=22 -ed_chm=22
# sample_map for dogs can be expt1, a, b, c
# ./Batch_scripts/build_labels.sh -gt=ancient -e=1 -sim -bl -n_o=3 -sm=time_block17H_3K -um=geo -st_chm=22 -ed_chm=22 -spt='100 100 100' -split='0.9 0.09 0.01'

Help()
{
    echo "This script builds labels in an unsupervised way to be used for regression later by Ge2Net"
    echo "Syntax: scriptTemplate [-gt|e|sm|ext|p_o|s|sim|bl|n_o|n_s|um|h]"
    echo "options:"
    echo "-gt|--geno_type       Specify the genotype as humans or dogs"
    echo "-sm|--sample_map        Specify the sample_map"
    echo "-e|--exp_id           Specify the experiment id"
    echo "-s|--seed             Specify the seed for splitting train/valid/test"
    echo "-ext|--extended_pca   Specify whether to compute pca or extended pca"
    echo "-p_o|--pop_order      If extended pca, specy the particular pop order as a list []"
    echo "-sim|--simulate         Specify whether to simulate admixed individuals"
    echo "-bl|--build_labels    Specify whether to build labels"
    echo "-n_o|--n_comp_overall  Specify the number of components for overall pca, typically 3"
    echo "-um|--unsupMethod      Specify which unsupervised method to use to create labels or use \"geo\" for geography "
    echo "-n_s|--n_comp_subclass Specify the number of components for subclasses for extended pca, typically 2"
    echo "-st_chm|--start_chm   Start chm for simulation"
    echo "-ed_chm|--end_chm     End chm for simulation"
    echo "-spt|--samples_per_type     Number of samples per generation to be taken for creating admixed. Twice this number(one each for maternal/paternal) will be created. Should be given as a [400, 400, 400] each for gen 2,4 and 8"
    echo "-split|--split_perc   train/val/test split. Should be given as [0.7,0.2,0.1]"
    echo "-gtr|--gens_to_ret    generations to return. Should be given as [2,4,8]"
    echo "-h|--help             Print this help manual"
    echo
}

for argument in "$@"; do 
    key=${argument%%=*}
    value=${argument#*=}

    echo "$key"
    echo "$value"
    echo "*******************"

    case "$key" in
        -gt|--geno_type )          geno_type=$value;;
        -sm|--sample_map )         sample_map=$value;;
        -e|--exp_id )              exp_id=$value;;
        -s|--seed )                seed=$value;;
        -ext|--extended_pca )      ext_pca="True";;
        -p_o|--pop_order )         pop_order=$value;;
        -sim|--simulate )          simulate="True";;
        -bl|--build_labels )       create_lbls="True";;
        -n_o|--n_comp_overall )     n_comp_overall=$value;;
        -n_s|--n_comp_subclass )    n_comp_subclass=$value;;
        -um|--unsupMethod )          unsupMethod=$value;;
        -vt|--vcf_type )            vcf_type=$value;;
        -st_chm|--start_chm )       start_chm=$value;;
        -ed_chm|--end_chm )         end_chm=$value;;
        -spt|--samples_per_type )     samples_per_type=$value;;
        -split|--split_perc )         split_perc=$value;;
        -gtr|--gens_to_ret )          gens_to_ret=$value;;
        -h |--help ) Help ; exit ;;
        \? ) echo "Error: Invalid option"; exit 1;;
    esac    
done

echo "Checking the validity of the arguments"

if [[ -z ${geno_type} ]]; then echo "Specify the geno_type with flag -gt|--geno_type"; exit 1; fi
if [[ -z ${exp_id} ]]; then echo "Specify the experiment id with flag -e|--exp_id"; exit 1; fi
if [[ -z ${ext_pca} && (${unsupMethod} = "pca") ]]; then 
echo "Setting extended_pca to False"
ext_pca="False"
n_comp_subclass=0
fi
if [[ -z ${pop_order} ]]; then echo "Setting pop order to default from the superpop dict"; fi
if [[ -z $simulate ]]; then echo "No admixture simulation will be performed"; simulate="False"; fi
if [[ -z ${create_lbls} ]]; then echo " no labels will be created"; create_lbls="False"; fi
if [[ -z ${n_comp_overall} ]]; then echo " overall number of components set to default of 3"; n_comp_overall=3; fi
if [[ (${ext_pca} = "True") && (-z ${n_comp_subclass})]]; then echo "subclass components set to default of 2"; n_comp_subclass=2; fi
if [[  (${ext_pca} = "False") && (${n_comp_subclass} >=0) && (${create_labels = "True"}) ]]; then echo "Invalid combination of extended pca and pca subclass components set"; exit 1; fi
if [[ ($simulate = "False") && (${create_lbls} = "False") ]]; then echo "Invalid combination of simulate and create labels set"; exit 1; fi
if [[ -z $seed ]]; then seed=$RANDOM; echo "seed not specified, setting to a random seed = $seed "; fi
if [[ (-z ${unsupMethod}) && (${create_lbls} = "True") ]]; then echo "No unsup method defined"; exit 1; fi
if [[ (-z ${sample_map}) && (${geno_type} = "dogs") ]]; then echo "No sample map specified"; exit 1; fi
if [[ (-z ${sample_map}) && (${geno_type} = "humans") ]]; then sample_map="None"; fi
if [[ -z $vcf_type ]] ; then echo "no specific vcf type specified"; fi
if [[ -z ${n_comp_subclass} ]]; then echo "setting n_comp_subclass to 0"; n_comp_subclass=0; fi
if [[ (${unsupMethod} = "geo") && (-z ${n_comp}) ]]; then echo "Setting n_comp=3 for n vectors"; n_comp=3; fi
if [[ -z ${samples_per_type} ]]; then echo "setting samples_per_type to default values of [400,400,400]"; samples_per_type=(400 400 400); fi
if [[ -z ${split_perc} ]]; then echo "setting split_perc to default values of [0.7,0.2,0.1]"; split_perc=(0.91 0.09 0.0); fi
if [[ -z ${gens_to_ret} ]]; then echo "setting gens_to_ret to None"; gens_to_ret=(2 4 8); fi

# set the vcf, genetic map and ref map according to genotype
echo "Setting variables for ${geno_type}"
if [[ (${geno_type} = 'humans') && (${vcf_type} != 'ukb') ]]; then
vcf_dir=$IN_PATH/${geno_type}/master_vcf_files/ref_final_beagle_phased_1kg_hgdp_sgdp_chr22.vcf.gz;
ref_map=$IN_PATH/${geno_type}/reference_files/reference_panel_metadata.tsv;
gen_map=$IN_PATH/${geno_type}/reference_files/allchrs.b38.gmap;
all_chm_snps=$OUT_PATH/${geno_type}/sm_${sample_map}/ld_False/all_chm_combined_snps_variance_filter_0.09_sample_win_0.npy;
# all_chm_snps=$OUT_PATH/${geno_type}/combined_chm/all_chm_combined_snps_variance_filter_0.3.npy;
n_comp=44; # smallest number of samples in a class is 44, only used for extended/residual pca
elif [[ (${geno_type} = 'humans') && (${vcf_type} = 'ukb') ]]; then
# vcf_dir=$IN_PATH/${geno_type}/${vcf_type}/filtered_references/ukb_snps_chm_1.recode.vcf;
vcf_filename=()
    start_chm=1;
    end_chm=22
    for chm in $(seq ${start_chm} ${end_chm})
        do
            vcf_filename+=($IN_PATH/${geno_type}/ukb/filtered_references/ukb_snps_chm_$chm.recode.vcf)
        done
vcf_dir=${vcf_filename[*]}
echo " vcf dir variable passed: ${vcf_dir}"
ref_map=$IN_PATH/${geno_type}/reference_files/reference_panel_metadata.tsv;
gen_map=$IN_PATH/${geno_type}/reference_files/allchrs.b38.gmap;
all_chm_snps=$OUT_PATH/${geno_type}/sm_${sample_map}/ld_0.5/vcf_type_${vcf_type}/all_chm_combined_snps_variance_filter_0.0_sample_win_0.npy
n_comp=44; # smallest number of samples in a class is 44, only used for extended/residual pca
elif [[  ${geno_type} = 'dogs' ]]; then
vcf_dir=$OUT_PATH/dogs/sm_${sample_map}/chr22/chr22_filtered.vcf.gz;
ref_map=$OUT_PATH/dogs/ref_map_${sample_map}.tsv;
gen_map=$IN_PATH/dogs/chr22/chr22_average_canFam3.1.txt;
all_chm_snps=$OUT_PATH/dogs/sm_${sample_map}/ld_False/vcf_type_/all_chm_combined_snps_variance_filter_0.0_sample_win_0.npy;
# all_chm_snps='$OUT_PATH/dogs/expt2_biallelic/all_chm_combined_snps_variance_filter_0.0_sample_win_100.npy';
n_comp=23; # smallest number of samples in a class is 23, only used for extended/residual pca
elif [[  ${geno_type} = 'ancient' ]]; then
vcf_dir=$OUT_PATH/${geno_type}/chr22/chr22_phased_imputed.vcf.gz;
ref_map=$OUT_PATH/${geno_type}/sm_${sample_map}.tsv;
gen_map=$OUT_PATH/${geno_type}/chr22/chr22_sm_coverage_2_re_ordered.map;
all_chm_snps="None"

else
echo "${geno_type} not supported"; exit 1 ;
fi

echo "Starting build_labels for geno type ${geno_type} experiment ${exp_id} for $unsupMethod and vcf_type ${vcf_type}"

if [[ -d $OUT_PATH/${geno_type}/labels/data_id_${exp_id}_$unsupMethod ]];
then
    echo " $OUT_PATH/${geno_type}/labels/data_id_${exp_id}_$unsupMethod already exists. Are you sure you want to overwrite ?";
    select yn in "Yes" "No"; do
        case $yn in
            Yes ) echo "okay going to overwrite and continue to start training"; break;;
            No ) echo "okay, exiting"; exit;;
        esac
    done
else
    echo "$OUT_PATH/${geno_type}/labels/data_id_${exp_id}_$unsupMethod doesn't exist, creating it";
    mkdir -p $OUT_PATH/${geno_type}/labels/data_id_${exp_id}_$unsupMethod
    echo "dir created"
fi


sbatch<<EOT
#!/bin/sh
#SBATCH -p bigmem
#SBATCH -c 1
#SBATCH --mem=1000G
#SBATCH -t 24:00:00
#SBATCH --output=$OUT_PATH/${geno_type}/labels/data_id_${exp_id}_$unsupMethod/log.out

ml load py-pytorch/1.4.0_py36
ml load py-scipy/1.4.1_py36
ml load py-numpy/1.18.1_py36
ml load py-matplotlib/3.2.1_py36
ml load py-pandas/1.0.3_py36

cd $USER_PATH
python3 buildLabels.py --data.seed $seed \
--data.expt_id ${exp_id} \
--data.reference_map ${ref_map} \
--data.sample_map ${sample_map} \
--data.vcf_dir ${vcf_dir[*]} \
--data.genetic_map ${gen_map} \
--data.geno_type ${geno_type} \
--data.extended_pca ${ext_pca} \
--data.simulate ${simulate} \
--data.create_labels ${create_lbls} \
--data.n_comp_overall ${n_comp_overall} \
--data.n_comp_subclass ${n_comp_subclass} \
--data.n_comp ${n_comp} \
--data.all_chm_snps ${all_chm_snps} \
--data.method ${unsupMethod} \
--data.start_chm ${start_chm} \
--data.end_chm ${end_chm} \
--data.samples_per_type ${samples_per_type[*]} \
--data.split_perc ${split_perc[*]} \
--data.gens_to_ret ${gens_to_ret[*]}
EOT

sleep .5
squeue -u richras
sleep .5
less +F $OUT_PATH/${geno_type}/labels/data_id_${exp_id}_$unsupMethod/log.out

# sample command from the terminal directly 
# cd /home/users/richras/Ge2Net_Repo
# source ini.sh
# python3 ./src/createLabels/buildLabels.py --data.seed 1234 --data.experiment_id 1 --data.referen
# ce_map $OUT_PATH/dogs/ref_map_expt1.tsv --data.sample_map expt1 --data.vcf_dir $OUT_PATH/dogs/sm_expt1/chr22/chr22_filtered.vcf --da
# ta.genetic_map $IN_PATH/dogs/chr22/chr22_average_canFam3.1.txt --data.geno_type dogs --data.extended_pca False --data.simulate True
# --data.create_labels True --data.n_comp_overall 3 --data.n_comp_subclass 0 --data.n_comp=23 --data.all_chm_snps $OUT_PATH/dogs/sm_ex
# pt1/ld_0.5/all_chm_combined_snps_variance_filter_0.0_sample_win_0.npy --data.method umap