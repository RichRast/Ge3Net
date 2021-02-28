import sys
import argparse
import os
import os.path as osp
from helper_funcs import Params

MODEL_CLASS = {'Model_A' : [['Model_A.model_A'],['AuxiliaryTask.AuxNetwork', 'BasicBlock.logits_Block']],
                    'Model_B' : [['Model_B.model_B'],['AuxiliaryTask.AuxNetwork', 'LSTM.BiRNN']],
                    'Model_C' : [['Model_C.model_C'],['AuxiliaryTask.AuxNetwork', 'LSTM.BiRNN']],
                    'Model_D' : [['Model_D.model_D'],['AuxiliaryTask.AuxNetwork', 'LSTM.BiRNN', 'BasicBlock.logits_Block']],
                    'Model_E' : [['Model_E.model_E'],['AuxiliaryTask.AuxNetwork', 'LSTM.BiRNN']],
                    'Model_F' : [['Model_F.Seq2Seq'],['AuxiliaryTask.AuxNetwork', 'Seq2Seq.Encoder', 'Seq2Seq.Decoder']],
                    'Model_G' : [['Model_G.model_G'],['AuxiliaryTask.AuxNetwork', 'LSTM.BiRNN']],
                    'Model_H' : [['Model_H.model_H'],['AuxiliaryTask.AuxNetwork', 'LSTM.BiRNN']],
                    'Model_I' : [['Model_I.model_I'],['AuxiliaryTask.AuxNetwork', 'Attention.attention', 'LSTM.BiRNN', 'BasicBlock.logits_Block']],
                    'Model_J' : [['Model_J.model_J'],['AuxiliaryTask.AuxNetwork', 'Conv.Conv1d']],
                    'Model_K' : [['Model_K.model_K'],['AuxiliaryTask.AuxNetwork', 'Attention.PositionalEncoding', 'Attention.attention_single', 'Attention.FFNN', 'BasicBlock.logits_Block']],
                    'Model_L' : [['Model_L.model_L'],['AuxiliaryTask.AuxNetwork', 'Attention.PositionalEncoding', 'Attention.attention_single', 'Attention.FFNN', 'Attention.attention_single', 'Attention.FFNN']],
                    'Model_M' : [['Model_M.model_M'],['AuxiliaryTask.AuxNetwork', 'Attention.PositionalEncoding', 'Transformer.transformer']],
                    'Model_N' : [['Model_N.model_N'],['AuxiliaryTask.AuxNetwork', 'LSTM.lstm', 'Attention.attention', 'LSTM.lstm']],
                    'Model_O' : [['Model_O.model_O'],['AuxiliaryTask.AuxNetwork', 'Attention.attention', 'LSTM.lstm', 'Attention.attention', 'LSTM.lstm']]
                    }   

parser = argparse.ArgumentParser()
parser.add_argument('--data.data_dir', type=str, default='/scratch/groups/cdbustam/richras', metavar='data_dir',
                        help='directory where simulated data is stored')
parser.add_argument('--data.params_dir', type=str, default='/home/users/richras/Ge2Net_Repo/experiments/pca/exp_D_1.12/', metavar='working_dir',
                    help='directory where json file for model hyperparameters are stored')
parser.add_argument('--data.labels_dir', type=str, default='/scratch/groups/cdbustam/richras/reference_files/pca_labels', metavar='pca_labels_dir',
                    help='pca labels built with maf 0.09')
parser.add_argument('--data.vcf_dir', type=str, default='/scratch/groups/cdbustam/richras/master_vcf_files/ref_final_beagle_phased_1kg_hgdp_sgdp_chr22.vcf.gz', metavar='vcf_dir',
                    help='directory where vcf file for the particular chm is saved')
parser.add_argument('--data.genetic_map', type=str, default='/scratch/groups/cdbustam/richras/reference_files/allchrs.b38.gmap', metavar='genetic_map_dir',
                    help='directory where genetic map is saved')
parser.add_argument('--data.reference_map', type=str, default='/scratch/groups/cdbustam/richras/reference_files/reference_panel_metadata.tsv', metavar='ref_file_path',
                help="reference sample map")
parser.add_argument('--data.experiment_id', type=int, default=1, metavar='experiment_id',
                    help='unique experiment identifier seed')
parser.add_argument('--data.experiment_name', type=str, default='unsupervised_labeling', metavar='experiment_name',
                    help='name of experiment')
parser.add_argument('--data.seed', type=int, default=1234, metavar='SEED',
                help='numpy seed')
#form labels
parser.add_argument('--data.form_labels', type=bool, default=True, metavar='form_labels',
                    help='set True if you want to form labels')
# admixed simulation
parser.add_argument('--data.simulate', type=bool, default=False, metavar='simulate_only',
                    help='set True if you want to simulate admixed data with ref indices')

parser.add_argument('--data.all_chm_snps', type=str, default='/scratch/groups/cdbustam/richras/combined_chm/all_chm_combined_snps_variance_filter_0.09.npy', metavar='all_chm',
                    help='numpy files with combined chm whose var>0.09')
parser.add_argument('--data.n_way', type=int, default=7, metavar='continent_way',
                    help='n_way classification for continent labels')
parser.add_argument('--data.samples_per_type', type=list, default=[400, 400, 400], metavar='num_samples',
                    help='num_samples per gen for each of train, valid and test')
parser.add_argument('--data.split_perc', type=list, default=[0.7, 0.2, 0.1], metavar='split_perc',
                    help='split percentage for train, valid and test')
parser.add_argument('--data.gens_to_ret', type=list, default=[2,4,8], metavar='gens_to_ret',
                    help='gens to simulate') 
# model arguments
parser.add_argument('--model.working_dir', type=str, default='/scratch/groups/cdbustam/richras/pca_models_dir', metavar='working_dir',
                    help='directory where models related to experiment is saved')
parser.add_argument('--model.pretrained', type=bool, default=False, metavar='pretrained',
                    help='specify whether to load pretrained model')
parser.add_argument('--model.pretrained_version', type=str, default='Model_L_1', metavar='pretrained_version',
                    help='specify the version of pretrained model')
parser.add_argument('--cuda', type=str, default='cuda:0', metavar='CUDA_DEVICE',
                    help='which cuda device to use')
# log args
parser.add_argument('--log.verbose', type=bool, default=True, metavar='verbose',
                    help='verbose')
parser.add_argument('--log.train', type=str, default='/scratch/groups/cdbustam/richras/logs/', metavar='log_train',
                    help='logging for training')
parser.add_argument('--log.valid', type=str, default='/scratch/groups/cdbustam/richras/logs/', metavar='log_valid',
                    help='logging for validation')
parser.add_argument('--log.wandb_dir', type=str, default='/scratch/users/richras/Batch_jobs/wandb', metavar='wandb_dir',
                    help='wandb logging dir')

def parse_args():
    # Load the parameters from json file
    args, unknown = parser.parse_known_args()
    config = vars(args)
    json_path = os.path.join(config['data.params_dir'], 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)
    
    # ToDo remove this later 
    os.environ["WANDB_SILENT"] = "true"
    os.environ["WANDB_DIR"] = config['log.wandb_dir']
    
    return config, params


# Todo: Too risky to automate this for now, after stress testing, this can be automated
# if params.hyper_search:
#     hyperparams_optimize.run(config, params)
# elif (config['data.simulate']==config['data.form_labels']==False):
#     trainer.run(config, params)
# else:
#     build_labels.run(config)
#     trainer.main(config, params)
