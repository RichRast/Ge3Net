import argparse
import os
import os.path as osp
import distutils.util
import logging

parser = argparse.ArgumentParser()                 
# model arguments
parser.add_argument('--data.params', type=str, required=False, help='directory where json file for model parameters are stored')
parser.add_argument('--data.dir', type=str, required=True, help='dir for data')
parser.add_argument('--data.labels', type=str, required=False, help='dir for labels')
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
