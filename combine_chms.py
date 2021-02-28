import numpy as np
import argparse
import os.path as osp
import sys

sys.path.insert(1, '/home/users/richras/GeNet_Repo')
from helper_funcs import vcf2npy, filter_snps, save_file
from visualization import plot_dist


parser = argparse.ArgumentParser(description='Create labels for training')
default_dataset = '1kg_hgdp_sgdp_chr22'
parser.add_argument('--data.data_dir', type=str, default='/scratch/users/richras/data/sim_data/pca/', metavar='data_dir',
                    help='directory where simulated data is stored')
parser.add_argument('--data.reference_map', type=str, default='/scratch/users/richras/admix/meta/meta/chm22/reference_panel_metadata.tsv', metavar='ref_file_path',
                help="reference sample map")
parser.add_argument('--data.vcf', type=str, default='/scratch/users/richras/data/ref_files/ref_final_beagle_phased_1kg_hgdp_sgdp_chr', metavar='vcf_file',
                    help="prefix for vcf file for each of the 22 chms")
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


def form_combined_chm(vcf_prefix, chm_start, chm_end, filter_thresh=None, verbose=True):
    
    chm = np.arange(chm_start, chm_end+1)
    
    for i in chm:
        vcf_file = vcf_prefix + str(i) + ".vcf.gz" 
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
            mean, std, filtered_snp_idx = filter_snps(mat_vcf_np, filter_thresh)
            if verbose:
                print(f' unfiltered snps for chm {i} is {mat_vcf_np.shape[1]}')
                print(f' filtered snps for chm {i} is {len(filtered_snp_idx)}')
                plot_dist(mean, std, i)
            if i==chm_start:
                filtered_vcf = mat_vcf_np[:,filtered_snp_idx]
            else:
                filtered_vcf = np.hstack((filtered_vcf, mat_vcf_np[:,filtered_snp_idx]))
    return filtered_vcf

def main(config):
    verbose_en = config['log.verbose']
    thresh = config['data.variance_filter']
    choose_k = config['data.choose_k']
    combined_snps = form_combined_chm(config['data.vcf'], config['data.chm_start'], \
        config['data.chm_end'],thresh)
    print(f'combined_snps shape:{combined_snps.shape}')
    combined_snps_save_path = osp.join(config['data.data_dir'], ('all_chm_combined_snps_maf_' + str(config['data.variance_filter']) + '.npy'))
    save_file(combined_snps_save_path, combined_snps)

if __name__=="__main__":
    config, unknown = parser.parse_known_args()
    config = vars(config)
    main(config)