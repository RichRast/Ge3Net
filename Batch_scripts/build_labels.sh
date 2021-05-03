#!/bin/bash
cd /home/users/richras/Ge2Net_Repo
source ini.sh

# sample command ./Batch_scripts/build_labels.sh -gt=dogs -e=14 -sim -bl -n_o=3 -sm=expt1 -s=1234 -um=umap
# sample_map for dogs can be expt1, a, b, c

Help()
{
    echo "This script builds labels in an unsupervised way to be used for regression later by Ge2Net"
    echo "Syntax: scriptTemplate [-gt|e|s|ext|p_o|s|bl|n_o|n_s|f|h]"
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
    echo "-n_s|--n_comp_subclass Specify the number of components for subclasses for extended pca, typically 2"
    echo "-f|--filter_criteria   Specify the filter criteria as either 'Single_Ancestry' or as 'rm_anc' "
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
        -f|--filter_criteria )     filter_criteria=$value;;
        -um|unsupMethod )          unsupMethod=$value;;
        -h | --help ) Help ; exit ;;
        \? ) echo "Error: Invalid option"; exit 1;;
    esac    
done

echo "Checking the validity of the arguments"

if [[ -z ${geno_type} ]]; then echo "Specify the geno_type with flag -gt|--geno_type"; exit 1; fi
if [[ -z ${exp_id} ]]; then echo "Specify the experiment id with flag -e|--exp_id"; exit 1; fi
if [[ -z ${ext_pca} ]]; then 
echo "Setting extended_pca to False"
ext_pca="False"
n_comp_subclass=0
fi
if [[ -z ${pop_order} ]]; then echo "Setting pop order to default from the superpop dict"; fi
if [[ -z $simulate ]]; then echo "No admixture simulation will be performed"; simulate="False"; fi
if [[ -z ${create_lbls} ]]; then echo " no labels will be created"; $create_lbls="False"; fi
if [[ -z ${n_comp_overall} ]]; then echo " overall pca components set to default of 3"; n_comp_overall=3; fi
if [[ (${ext_pca} = "True") && (-z ${n_comp_subclass})]]; then echo "subclass components set to default of 2"; n_comp_subclass=2; fi
if [[  (${ext_pca} = "False") && (${n_comp_subclass} >=0) ]]; then echo "Invalid combination of extended pca and pca subclass components set"; exit 1; fi
if [[ ($simulate = "False") && (${create_lbls} = "False") ]]; then echo "Invalid combination of simulate and create labels set"; exit 1; fi
if [[ -z $seed ]]; then seed=$RANDOM; echo "seed not specified, setting to a random seed = $seed "; fi
if [[ (-z ${unsupMethod}) && (${create_lbls} = "True") ]]; then echo "No unsup method defined"; exit 1; fi
if [[ (-z ${sample_map}) ]]; then echo "No sample map specified"; exit 1; fi

# set the vcf, genetic map and ref map according to genotype
echo "Setting variables for ${geno_type}"
if [[ ${geno_type} = 'humans' ]]; then
vcf_dir=$IN_PATH/${geno_type}/master_vcf_files/ref_final_beagle_phased_1kg_hgdp_sgdp_chr22.vcf.gz;
ref_map=$IN_PATH/${geno_type}/reference_files/reference_panel_metadata.tsv;
gen_map=$IN_PATH/${geno_type}/reference_files/allchrs.b38.gmap;
filter_criteria='Single_Ancestry';
all_chm_snps=$OUT_PATH/${geno_type}/combined_chm/all_chm_combined_snps_variance_filter_0.3.npy;
n_comp=44;
elif [[  ${geno_type} = 'dogs' ]]; then
vcf_dir=$OUT_PATH/dogs/sm_${sample_map}/chr22/chr22_filtered.vcf;
ref_map=$OUT_PATH/dogs/ref_map_${sample_map}.tsv;
gen_map=$IN_PATH/dogs/chr22/chr22_average_canFam3.1.txt;
filter_criteria='';
all_chm_snps=$OUT_PATH/dogs/sm_${sample_map}/ld_0.5/all_chm_combined_snps_variance_filter_0.0_sample_win_0.npy;
# all_chm_snps='$OUT_PATH/dogs/expt2_biallelic/all_chm_combined_snps_variance_filter_0.0_sample_win_100.npy';
n_comp=23;
else
echo "${geno_type} not supported"; exit 1 ;
fi

echo "Starting build_labels for geno type ${geno_type} experiment ${exp_id}"
mkdir -p $OUT_PATH/${geno_type}/unsupervised_labels/sm_${sample_map}/${exp_id}

sbatch<<EOT
#!/bin/sh
#SBATCH -p bigmem
#SBATCH -c 1
#SBATCH --mem=1000G
#SBATCH -t 24:00:00
#SBATCH --output=$OUT_PATH/build_labels_gt_${geno_type}_sm_${sample_map}_exp_${exp_id}_seed_$seed.out

ml load py-pytorch/1.4.0_py36
ml load py-scipy/1.4.1_py36
ml load py-numpy/1.18.1_py36
ml load py-matplotlib/3.2.1_py36
ml load py-pandas/1.0.3_py36

cd /home/users/richras/Ge2Net_Repo
python3 buildLabels.py --data.seed $seed \
--data.experiment_id ${exp_id} \
--data.reference_map ${ref_map} \
--data.sample_map ${sample_map} \
--data.vcf_dir ${vcf_dir} \
--data.genetic_map ${gen_map} \
--data.geno_type ${geno_type} \
--data.extended_pca ${ext_pca} \
--data.simulate ${simulate} \
--data.create_labels ${create_lbls} \
--data.n_comp_overall ${n_comp_overall} \
--data.n_comp_subclass ${n_comp_subclass} \
--data.n_comp ${n_comp} \
--data.all_chm_snps ${all_chm_snps} \
--data.method ${unsupMethod}
EOT

sleep .5
squeue -u richras
sleep .5
less +F $OUT_PATH/build_labels_gt_${geno_type}_sm_${sample_map}_exp_${exp_id}_seed_$seed.out