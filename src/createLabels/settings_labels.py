import sys
import argparse
import os
import os.path as osp
import distutils.util

parser = argparse.ArgumentParser()                 
parser.add_argument('--data.geno_type', type=str, required=True, choices=['humans', 'dogs', 'ancient'], help="genotype of humans or dogs")
parser.add_argument('--data.vcf_dir', nargs="*", help='directory where vcf file for the particular chm is saved')
parser.add_argument('--data.genetic_map', type=str, default=osp.join( os.environ.get('IN_PATH'), 'reference_files/allchrs.b38.gmap'), help='directory where genetic map is saved')
parser.add_argument('--data.sample_map', type=str, required=False, help="type of sample map such as expt1, a, b c or keep")
parser.add_argument('--data.reference_map', type=str, default=osp.join(os.environ.get('IN_PATH'), 'reference_files/reference_panel_metadata.tsv'), help="reference sample map")
parser.add_argument('--data.expt_id', type=str, required=True, help='unique experiment identifier seed')
parser.add_argument('--data.seed', type=int, default=1234, metavar='SEED',help='numpy seed')
parser.add_argument('--data.pop_order', type=str, required=False, help='directory for a specific pop order')
parser.add_argument('--data.method', type=str, required=False, default='pca', help='method to create labels')
parser.add_argument('--data.n_comp_overall', type=int, required=False, help='overall components of pca')
parser.add_argument('--data.n_comp_subclass', type=int, required=False, help='subclass components of extended pca')
parser.add_argument('--data.n_comp', type=int, required=False, help='total number of components (relevant for extended pca)')

# build labels
parser.add_argument('--data.create_labels', type=distutils.util.strtobool, default='False', help='set True if you want to form labels')

# admixed simulation
parser.add_argument('--data.simulate', type=distutils.util.strtobool, default='True', help='set True if you want to simulate admixed data with ref indices')
parser.add_argument('--data.all_chm_snps', type=str, required=False, default=osp.join(os.environ.get('IN_PATH'), \
    'combined_chm/all_chm_combined_snps_variance_filter_0.3.npy'), help='numpy files with combined chm whose var>0.09')
parser.add_argument('--data.samples_per_type', type=list, default=[400, 400, 400], help='num_samples per gen for each of train, valid and test')
parser.add_argument('--data.split_perc', type=list, default=[0.7, 0.2, 0.1], help='split percentage for train, valid and test')
parser.add_argument('--data.gens_to_ret', type=list, default=[2,4,8], help='gens to simulate') 
parser.add_argument('--data.start_chm', type=int, required=False, help='start chm if doing admixture combined')
parser.add_argument('--data.end_chm', type=int, required=False, help='end chm if doing admixture combined')

# log args
parser.add_argument('--log.verbose', type=distutils.util.strtobool, default='True', help='verbose')
parser.add_argument('--log.dir', type=str, help='logging directory')
parser.add_argument('--log.wandb_dir', type=str, default=osp.join( os.environ.get('USER_SCRATCH_PATH'), 'Batch_jobs/wandb'), help='wandb logging dir')

def parse_args():
    # Load the parameters from json file
    args, unknown = parser.parse_known_args()
    config = vars(args)
    
    # ToDo remove this later 
    os.environ["WANDB_SILENT"] = "true"
    os.environ["WANDB_DIR"] = config['log.wandb_dir']
    
    return config
