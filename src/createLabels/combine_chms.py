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
parser.add_argument('--data.vcf_filenames', nargs="*",
                help="name of vcf files")
parser.add_argument('--data.save_path', type=str, metavar='save_path',
                help="path where the output is saved")
parser.add_argument('--data.variance_filter', type=float, default=0.09, metavar='filter_threshold',
                    help="variance filter threshold for filtering snps")
parser.add_argument('--data.choose_k', type=float, default=None, metavar='choose_k_snps',
                    help="choose the top k snps with the most variance")
parser.add_argument('--data.sample_win', type=int, default=0, metavar='sample_win',
                    help="subsample window size")
parser.add_argument('--log.verbose', type=bool, default=True, metavar='verbose',
                    help='verbose')


def process_filter_chm(vcf_filenames, save_path, combine, sample_win, filter_thresh=None, verbose=True):
    
    for i, vcf_file in enumerate(vcf_filenames):
        #parse the chm # from the vcf filename
        print(f'vcf_file:{vcf_file}')
        chm_tmp = vcf_file.split('chr')[-1]
        print(f'chm_tmp:{chm_tmp}')
        if '_' in chm_tmp: chm = chm_tmp.split('_')[0]
        else: chm = chm_tmp.split('.')[0]
        print(f'chm:{chm}')
        if combine:
            mat_vcf_np = vcf2npy(vcf_file) 
            # subsample 1 out of every x snps 
            print(f'mat_vcf_np shape before subsample:{mat_vcf_np.shape}')
            mat_vcf_np=mat_vcf_np[:,1::sample_win] if sample_win>0 else mat_vcf_np
            print(f'mat_vcf_np shape after subsample:{mat_vcf_np.shape}')
            if filter_thresh == 0.0:
                print("combining all chms with no threshold")
                if i==0:
                    filtered_vcf = mat_vcf_np
                else:
                    filtered_vcf = np.hstack((filtered_vcf, mat_vcf_np))
                print(f'finished combining chm {chm}')
            
            elif filter_thresh == "combined_variance":
                print("combining the variance of all chms")
                if i==0:
                    filtered_vcf = mat_vcf_np.var(axis=0)
                else:
                    filtered_vcf = np.hstack((filtered_vcf, mat_vcf_np.var(axis=0)))
                print(f'finished combining chm {chm} and filtered vcf shape is {filtered_vcf.shape}')
            
            else:
                mean, var, filtered_snp_idx = filter_snps(mat_vcf_np, filter_thresh)
                if verbose:
                    print(f' unfiltered snps for chm {chm} is {mat_vcf_np.shape[1]}')
                    print(f' mean of snps for chm {chm} is {mean}')
                    print(f' var of snps for chm {chm} is {var}')
                    print(f' filtered snps for chm {chm} is {len(filtered_snp_idx)}')
                    plot_dist(mean, var, chm)
                if i==0:
                    filtered_vcf = mat_vcf_np[:,filtered_snp_idx]
                else:
                    filtered_vcf = np.hstack((filtered_vcf, mat_vcf_np[:,filtered_snp_idx]))
        else:
            filtered_vcf = filter_vcf(vcf_file, filter_thresh)
            #save each filtered vcf file
            filtered_save_filename = ''.join([save_path, '/filtered_var_', str(filter_thresh) ,'chm_', str(chm), '_sample_win_', str(sample_win),'.vcf.gz'])
            print(f"saving filtered vcf for chm {chm} at {str(filtered_save_filename)} \n")
            save_file(filtered_save_filename, filtered_vcf, en_pickle=True)
            
    # if combine, then save the combined snps npy 
    if combine:
        print(f'combined_snps shape:{filtered_vcf.shape}')
        combined_snps_save_filename = ''.join([save_path, '/all_chm_combined_snps_variance_filter_', str(filter_thresh), '_sample_win_', str(sample_win), '.npy'])
        print(f"saving combined filtered vcf at {str(combined_snps_save_filename)}")
        save_file(combined_snps_save_filename, filtered_vcf) 
    
def main(config):
    # print the configurations in the log directory
    for k, v in config.items():
        print(f"config for {k} : {v}")

    verbose_en = config['log.verbose']
    thresh = config['data.variance_filter']
    # choose_k = config['data.choose_k']
    
    process_filter_chm(config['data.vcf_filenames'], config['data.save_path'], config['data.combine'], config['data.sample_win'], thresh)

if __name__=="__main__":
    config, unknown = parser.parse_known_args()
    config = vars(config)
    main(config)