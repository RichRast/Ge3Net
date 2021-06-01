import argparse
import os
import os.path as osp
import distutils.util

MODEL_CLASS = {'Model_A' : [['Model_A.model_A'],['AuxiliaryTask.AuxNetwork', 'BasicBlock.logits_Block']],
                    'Model_B' : [['Model_B.model_B'],['AuxiliaryTask.AuxNetwork', 'LSTM.BiRNN', 'BasicBlock.logits_Block']],
                    'Model_C' : [['Model_C.model_C'],['AuxiliaryTask.AuxNetwork', 'LSTM.BiRNN', 'BasicBlock.logits_Block']],
                    'Model_D' : [['Model_D.model_D'],['AuxiliaryTask.BaseNetwork', 'LSTM.BiRNN', 'BasicBlock.logits_Block']],
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
# model arguments
parser.add_argument('--data.params', type=str, required=False, help='directory where json file for model parameters are stored')
parser.add_argument('--data.dir', type=str, required=True, help='dir for data')
parser.add_argument('--data.labels', type=str, required=False, help='dir for labels')
parser.add_argument('--model.pretrained', type=distutils.util.strtobool, default='False',help='specify whether to load pretrained model')
parser.add_argument('--model.pretrained_dir', type=str, required=False, help='specify the dir of pretrained model (used for testing/inference)')
parser.add_argument('--model.loadBest', type=distutils.util.strtobool, default="True", help='load the best or last checkpoint saved')
parser.add_argument('--cuda', type=str, required=False, default='cuda', help='which cuda device to use')
parser.add_argument('--models.dir', type=str, required=True, help='model logging directory')
parser.add_argument('--model.summary', type=str, default='default', help='description of what makes this model run special')
parser.add_argument('--data.dataset_type', type=str, required=False, default='valid', choices=['train', 'valid', 'test'], help='used for testing/inference')
# log args
parser.add_argument('--log.verbose', type=distutils.util.strtobool, default='True', help='verbose')
parser.add_argument('--log.wandb_dir', type=str, default=osp.join( os.environ.get('USER_SCRATCH_PATH'), 'Batch_jobs/wandb'), help='wandb logging dir')

def parse_args():
    # Load the parameters from json file
    args, unknown = parser.parse_known_args()
    config = vars(args)
    
    # ToDo remove this later 
    os.environ["WANDB_SILENT"] = "true"
    os.environ["WANDB_DIR"] = config['log.wandb_dir']
    
    return config

