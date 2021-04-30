import sys
import argparse
import os
import os.path as osp
import logging
import copy

from models import LSTM, AuxiliaryTask, Conv, Attention, Transformer, BasicBlock, Model_A, Model_B, Model_C, Model_D, Model_E, \
Model_F, Seq2Seq, Model_G, Model_H, Model_I, Model_J, Model_K, Model_L, Model_M, Model_N, Model_O
from utils import set_logger, load_model, Params
from dataset import Haplotype
from settings import parse_args, MODEL_CLASS

import torch
import numpy as np
from collections import namedtuple
import wandb
from decorators import timer

@timer
def main(config, params):
    # use GPU if available
    params.cuda = torch.cuda.is_available()

    # Set the random seed for reproducible experiments
    SEED = params.train_seed
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    if params.cuda:
        torch.cuda.manual_seed(SEED)
        torch.backends.cudnn.deterministic = True
        # cudnn benchmark enabled for memory space optimization
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        
    if config['log.verbose']:
        # Set the logger
        set_logger(osp.join(config['log.train'], config['model_version']))
        # use the major version only
        wandb.init(project=''.join([str(params.model), '_', str(params.major_version)]), config=params)
        params=wandb.config

    # configure device
    params.device = torch.device(config['cuda'] if params.cuda else 'cpu')

    # Create the input data pipeline
    logging.info("Loading the datasets...")

    dataset_path = osp.join(str(config['data.data_out']), config['data.experiment_name'], str(config['data.experiment_id']))
    test_dataset = Haplotype('no_label', dataset_path, params)
    
    test_generator = torch.utils.data.DataLoader(test_dataset, batch_size=params.batch_size, shuffle=True,
                                                    num_workers=0)
     
    # Create the model
    model_subclass, model_basics = MODEL_CLASS[params.model]
    
    model_params =[]
    middle_models=[]
    
    for i, model_basic in enumerate(model_basics):
        # instantiate the model class
        m = eval(model_basic)(params).to(params.device)
        middle_models.append(m)
        print(f'model {model_subclass} : {model_basic}')
            
    model_path = osp.join(config['model.working_dir'], config['model.pretrained_version'], 'best.pt')
    model_ret = load_model(model_path, middle_models)

    # call to the corresponding model, example - Model_L.model_L
    model = eval(model_subclass[0])(*model_ret, params=params)

    test_result = model.pred(test_generator, wandb=wandb)
    
    return test_result
    
if __name__=="__main__":
    config, params = parse_args()
    config['model_version'] = ''.join([str(params.model), '_', str(params.major_version), '.', str(params.minor_version)])
    main(config, params)
