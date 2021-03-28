import sys
import argparse
import os
import os.path as osp
import logging
import copy

from models import LSTM, AuxiliaryTask, Conv, Attention, Transformer, BasicBlock, Model_A, Model_B, Model_C, Model_D, Model_E, \
Model_F, Seq2Seq, Model_G, Model_H, Model_I, Model_J, Model_K, Model_L, Model_M, Model_N, Model_O
from helper_funcs import save_checkpoint, set_logger, load_model, early_stopping, Params, weight_int, custom_opt, load_path
from dataset import Haplotype
from settings import parse_args, MODEL_CLASS
from visualization import Plot_per_epoch_revised
from build_labels_revised import repeat_pop_arr
import matplotlib.pyplot as plt

import torch
import numpy as np
import pandas as pd
from collections import namedtuple
from torch.utils.tensorboard import SummaryWriter
import wandb
from decorators import timer

@timer
def main(config, params, trial=None):
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
        
        # Tensorboard Writer
        #writer_train = SummaryWriter(config['log.train'])
        #writer_val = SummaryWriter(config['log.valid'])

    # configure device
    params.device = torch.device(config['cuda'] if params.cuda else 'cpu')

    # Create the input data pipeline
    logging.info("Loading the datasets...")

    dataset_path = osp.join(str(config['data.data_out']), config['data.geno_type'], config['data.experiment_name'], str(config['data.experiment_id']))
    labels_path = config['data.labels_dir']
    training_dataset = Haplotype('train', dataset_path, params, labels_path, config['data.geno_type'])
    validation_dataset = Haplotype('valid', dataset_path, params, labels_path, config['data.geno_type'])
    
    training_generator = torch.utils.data.DataLoader(training_dataset, batch_size=params.batch_size, shuffle=True,
                                                    num_workers=0)
    validation_generator = torch.utils.data.DataLoader(validation_dataset, batch_size=params.batch_size, num_workers=0)
     
    # Initiate the class for plotting per epoch
    if params.plotting:
        if config['data.geno_type']=='humans':
            pop_dict = load_path(osp.join(dataset_path, 'granular_pop.pkl'), en_pickle=True)
        elif config['data.geno_type']=='dogs':
            pop_dict = load_path(osp.join(dataset_path, 'superpop.pkl'), en_pickle=True)
        rev_pop_dict = {v:k for k,v in pop_dict.items()}
        pop_sample_map = pd.read_csv(osp.join(labels_path, params.pop_sample_map), sep='\t')
        pop_arr = repeat_pop_arr(pop_sample_map, config['data.geno_type'])
        plot_obj = Plot_per_epoch_revised(params.n_comp_overall, params.n_comp_subclass, params.pop_num, rev_pop_dict, pop_arr)
        
         
    # Create the model
    model_subclass, model_basics = MODEL_CLASS[params.model]
    
    model_params =[]
    middle_models=[]
    
    for i, model_basic in enumerate(model_basics):
        params_dict={}
        # instantiate the model class
        m = eval(model_basic)(params).to(params.device)
        middle_models.append(m)
        print(f'model {model_subclass} : {model_basic}')
        # assign the corresponding model parameters
        params_dict['params']= m.parameters()
        params_dict['lr'] = params.learning_rate[i]
        params_dict['weight_decay'] = params.weight_decay[i]
        model_params.append(params_dict)
    
    # initialize the weights and biases of models
    for m in middle_models:
        m.apply(weight_int)
        
    if config['log.verbose']:
        #watch all the models
        for m in middle_models:
            wandb.log({"model_name":str(m)})
            wandb.watch(m, log='all')
    
    # call to the corresponding model, example - Model_L.model_L
    model = eval(model_subclass[0])(*middle_models, params=params)

    if config['model.pretrained']:
        model_path = osp.join(config['model.working_dir'], config['model.pretrained_version'], 'best.pt')
        #model_main, model_aux, start_epoch, optimizer = load_model(model_path, model_aux, model_main, optimizer)

    training_loop(model, model_params, middle_models, params, config, training_generator, validation_generator, plot_obj, wandb)    
   
    
@timer
def training_loop(model, model_params, middle_models, params, config, training_generator, validation_generator, plot_obj, wandb):
     # optimizer
    optimizer = torch.optim.Adam(model_params)
    #custom_optimizer = custom_opt(optimizer, d_model=params.att['input_size'], warmup_steps=params.att['warmup_steps'], factor=params.att['factor'], groups=params.lr_groups)
    
    # learning rate scheduler
    exp_lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience = params.lr_steps_decay, verbose=True)

    logging.info("Begin Training....")
    start_epoch = 0
    patience = 0

    for epoch in range(start_epoch, params.num_epochs):
        train_result = model.train(optimizer, training_generator, plot_obj, wandb=wandb)

        eval_result = model.valid(validation_generator, plot_obj, wandb=wandb)
        plt.close('all')
        
        if config['log.verbose']:
            #writer_train.add_scalar('Accuracy/train', train_result.accr.weighted_loss, epoch)
            #writer_val.add_scalar('Accuracy/val', eval_result.accr.weighted_loss, epoch)

            #writer_train.flush()
            #writer_val.flush()

            wandb.log({"train_metrics":train_result.accr._asdict(), "epoch":epoch})
            wandb.log({"valid_metrics":eval_result.accr._asdict(), "epoch":epoch})
            eval_result_dict = eval_result.accr._asdict()
            wandb.log({"valid_weighted_loss":eval_result_dict["l1_loss"], "epoch": epoch})

        # every step in the scheduler is per epoch
        exp_lr_scheduler.step(eval_result.accr.weighted_loss)
        
        # logic for best model
        is_best = False
        if (epoch==start_epoch) or (eval_result.accr.weighted_loss < best_val_accr):
            best_val_accr = eval_result.accr.weighted_loss
            is_best = True
        
        if epoch!=start_epoch:
            patience = early_stopping(eval_result.accr.weighted_loss, val_prev_accr, patience, params.thresh)
            if patience == params.early_stopping_thresh:
                logging.info("Early stopping...")
                break
        
        val_prev_accr = eval_result.accr.weighted_loss

        # saving a model at every epoch
        logging.info(f"Saving at epoch {epoch}")
        logging.info(f'train accr: {train_result.accr.weighted_loss}, val accr: {eval_result.accr.weighted_loss}')
        checkpoint = osp.join(config['model.working_dir'], config['model_version'])
        models_state_dict = [middle_models[i].state_dict() for i in range(len(middle_models))]

        save_checkpoint({
            'epoch': epoch,
            'model_state_dict': models_state_dict,
            'optimizer_state_dict': optimizer.state_dict(),
            'accr': eval_result._asdict(),
            'train_accr': train_result._asdict()
            }, checkpoint, is_best=is_best)
        
        if params.hyper_search_type=='optuna':    
            trial.report(eval_result.accr.weighted_loss, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
            
            return best_val_accr
        

    del train_result, eval_result, model, middle_models, model_params, \
        training_generator, validation_generator
    torch.cuda.empty_cache()

    
if __name__=="__main__":
    config, params = parse_args()
    config['model_version'] = ''.join([str(params.model), '_', str(params.major_version), \
        '.', str(params.minor_version), '.', str(config['model.expt_id'])])
    main(config, params)
