import numpy as np
import argparse
import os
import os.path as osp
import sys
import distutils.util 

sys.path.insert(1, '/home/users/richras/GeNet_Repo')
from helper_funcs import vcf2npy, filter_snps, save_file, filter_vcf
from visualization import plot_dist


parser = argparse.ArgumentParser(description='Create labels for training')
default_dataset = '1kg_hgdp_sgdp_chr22'
parser.add_argument('--data.combine', type=distutils.util.strtobool, default='False', 
                    help='flag to indicate whether to combine the snps across chm')
parser.add_argument('--data.geno_type', type=str, default='humans', metavar='genotype',
                help="genotype of humans or dogs")
parser.add_argument('--data.variance_filter', type=float, default=0.09, metavar='filter_threshold',
                    help="variance filter threshold for filtering snps")
parser.add_argument('--data.choose_k', type=float, default=None, metavar='choose_k_snps',
                    help="choose the top k snps with the most variance")
parser.add_argument('--data.chm_start', type=int, default=1, metavar='first_chm',
                    help="first chm to combine")
parser.add_argument('--data.chm_end', type=int, default=22, metavar='last_chm',
                    help="last chm to combine")

parser.add_argument('--log.verbose', type=bool, default=True, metavar='verbose',
                    help='verbose')


def process_filter_chm(geno_type, vcf_prefix, chm_start, chm_end, filtered_save_path, combine, filter_thresh=None, verbose=True):
    
    chm = np.arange(chm_start, chm_end+1)

    for i in chm:
        if geno_type=='humans':
            vcf_file = ''.join([vcf_prefix, '/master_vcf_files/ref_final_beagle_phased_1kg_hgdp_sgdp_chr', str(i), ".vcf.gz"])
        elif geno_type=='dogs':
            vcf_file = ''.join([vcf_prefix, '/chr', str(i), '/', 'chr', str(i), '_expt1_filtered.vcf.gz'])

        if combine:
            mat_vcf_np = vcf2npy(vcf_file) 
            if filter_thresh is None:
                print("combining all chms with no threshold")
                if i==chm_start:
                    filtered_vcf = mat_vcf_np
                else:
                    filtered_vcf = np.hstack((filtered_vcf, mat_vcf_np))
                print(f'finished combining chm {i}')
            
            elif filter_thresh == 0.0:
                print("combining the variance of all chms")
                if i==chm_start:
                    filtered_vcf = mat_vcf_np.var(axis=0)
                else:
                    filtered_vcf = np.hstack((filtered_vcf, mat_vcf_np.var(axis=0)))
                print(f'finished combining chm {i} and filtered vcf shape is {filtered_vcf.shape}')
            
            else:
                mean, var, filtered_snp_idx = filter_snps(mat_vcf_np, filter_thresh)
                if verbose:
                    print(f' unfiltered snps for chm {i} is {mat_vcf_np.shape[1]}')
                    print(f' mean of snps for chm {i} is {mean}')
                    print(f' var of snps for chm {i} is {var}')
                    print(f' filtered snps for chm {i} is {len(filtered_snp_idx)}')
                    plot_dist(mean, var, i)
                if i==chm_start:
                    filtered_vcf = mat_vcf_np[:,filtered_snp_idx]
                else:
                    filtered_vcf = np.hstack((filtered_vcf, mat_vcf_np[:,filtered_snp_idx]))
        else:
            filtered_vcf = filter_vcf(vcf_file, filter_thresh)
            #save each filtered vcf file
            filtered_save_filename = ''.join([filtered_save_path, '/filtered_', 'chm_', str(i), '.vcf'])
            print(f"saving filtered vcf for chm {i} at {str(filtered_save_filename)} \n")
            save_file(filtered_save_filename, filtered_vcf, en_pickle=True)
            
    # if combine, then save the combined snps npy 
    if combine:
        print(f'combined_snps shape:{filtered_vcf.shape}')
        combined_snps_save_filename = ''.join([filtered_save_path, '/all_chm_combined_snps_variance_filter_', str(config['data.variance_filter']), '.npy'])
        print(f"saving combined filtered vcf at {str(combined_snps_save_filename)}")
        save_file(combined_snps_save_filename, filtered_vcf) 
    
def main(config):
    verbose_en = config['log.verbose']
    thresh = config['data.variance_filter']
    # choose_k = config['data.choose_k']

    vcf_prefix = osp.join(os.environ.get('IN_PATH'), config['data.geno_type'])

    filtered_save_path = osp.join(os.environ.get('OUT_PATH'), config['data.geno_type'], ''.join(['filtered_var_', str(thresh)]))
    if not osp.exists(filtered_save_path):
        os.mkdir(filtered_save_path)

    process_filter_chm(config['data.geno_type'], vcf_prefix, config['data.chm_start'], \
        config['data.chm_end'], filtered_save_path, config['data.combine'], thresh)


if __name__=="__main__":
    config, unknown = parser.parse_known_args()
    config = vars(config)
    main(config)