import os
import os.path as osp
import logging
import sys
from src.models import LSTM, AuxiliaryTask, Conv, Attention, Transformer, BasicBlock, Model_A, \
Model_B, Model_C
# Model_D, Model_E, \
# Model_F, Seq2Seq, Model_G, Model_H, Model_I, Model_J, Model_K, Model_L, Model_M, Model_N, Model_O
from src.utils.modelUtil import save_checkpoint, load_model, early_stopping, Params,\
     weight_int, custom_opt, CustomDataParallel
from src.utils.dataUtil import set_logger, load_path
from src.utils.labelUtil import repeat_pop_arr
from src.utils.decorators import timer
from src.main.dataset import Haplotype
from src.main.settings_model import parse_args, MODEL_CLASS
from src.main.visualization import Plot_per_epoch
import matplotlib.pyplot as plt

import torch
import numpy as np
import pandas as pd
import wandb

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
        set_logger(config['log.dir'])
        # use the major version only
        wandb.init(project=''.join([str(params.model)]), config=params, allow_val_change=True)
        # params=wandb.config
        
    # configure device
    params.device = torch.device(config['cuda'] if params.cuda else 'cpu')

    #============================= Create and load datasets ===============================#
    # Create the input data pipeline
    logging.info("Loading the datasets...")

    labels_path = config['data.labels_dir']
    training_dataset = Haplotype('train', params, labels_path)
    validation_dataset = Haplotype('valid', params, labels_path)
    test_dataset = Haplotype('test', params, labels_path)
    
    training_generator = torch.utils.data.DataLoader(training_dataset, batch_size=params.batch_size, shuffle=True,
                                                    num_workers=0)
    validation_generator = torch.utils.data.DataLoader(validation_dataset, batch_size=params.batch_size, num_workers=0)

    test_generator = torch.utils.data.DataLoader(test_dataset, batch_size=params.batch_size, num_workers=0)
     
    # Initiate the class for plotting per epoch
    plotObj=None
    if params.plotting:
        pop_dict = load_path(osp.join(labels_path, 'granular_pop.pkl'), en_pickle=True)
        rev_pop_dict = {v:k for k,v in pop_dict.items()}
        pop_sample_map = pd.read_csv(osp.join(labels_path, params.pop_sample_map), sep='\t')
        pop_arr = repeat_pop_arr(pop_sample_map)
        plotObj = Plot_per_epoch(params.n_comp_overall, params.n_comp_subclass, \
            rev_pop_dict, pop_arr, pop_order=params.pop_num, geography=params.geography)
        
    #============================= Create and load the model ===============================#    
    # Create the model
    model_subclass, model_basics = MODEL_CLASS[params.model]
    
    model_params =[]
    middle_models=[]
    
    for i, model_basic in enumerate(model_basics):
        params_dict={}
        # instantiate the model class
        # if not params.dict.get(model_basic):
        #     continue
        m = eval(model_basic)(params)
        # use parallel GPU's
        if torch.cuda.device_count() > 1:
            logging.info("Using", torch.cuda.device_count(), "GPUs")
            m = torch.nn.DataParallel(m, dim=0)
        m.to(params.device)
        logging.info(f"is the model on cuda? : {next(m.parameters()).is_cuda}")
        middle_models.append(m)
        logging.info(f'model {model_subclass} : {model_basic}')
        # assign the corresponding model parameters
        params_dict['params']= m.parameters()
        params_dict['lr'] = params.learning_rate[i]
        params_dict['weight_decay'] = params.weight_decay[i]
        model_params.append(params_dict)
    
    # initialize the weights and biases of models
    for m in middle_models:
        m.apply(weight_int)
        
    # Total number of parameters

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

    training_loop(model, model_params, middle_models, params, config, training_generator, \
        validation_generator, test_generator, plotObj, wandb)    
   
def epoch_logger(wandb, phase, result, epoch, geography, cp_predict, superpop_predict):
    wandb.log({f"{phase}_metrics":result.t_accr._asdict(), "epoch":epoch})
    if geography:
        wandb.log({f"{phase}_metrics":result.t_balanced_gcd._asdict(), "epoch":epoch})
    if cp_predict:
        wandb.log({f"{phase}_metrics":result.t_cp_accr._asdict(), "epoch":epoch})
    if superpop_predict:
        wandb.log({f"{phase}_metrics":result.t_sp_accr._asdict(), "epoch":epoch})
    
@timer
def training_loop(model, model_params, middle_models, params, config, training_generator, validation_generator, \
    test_generator, plotObj, wandb):
     # optimizer
    optimizer = torch.optim.Adam(model_params)
    #custom_optimizer = custom_opt(optimizer, d_model=params.att['input_size'], \
    # warmup_steps=params.att['warmup_steps'], factor=params.att['factor'], groups=params.lr_groups)
    
    # learning rate scheduler
    exp_lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience = params.lr_steps_decay, verbose=True)

    logging.info("Begin Training....")
    start_epoch = 0
    patience = 0

    for epoch in range(start_epoch, params.num_epochs):
        train_result = model.train(optimizer, training_generator, plotObj=plotObj, wandb=wandb)

        eval_result = model.valid(validation_generator, plotObj=plotObj, wandb=wandb)
        plt.close('all')
        
        # Todo user partial here
        if wandb is not None:
            epoch_logger(wandb, "train", train_result, epoch, params.geography, params.cp_predict, params.superpop_predict)
            epoch_logger(wandb, "valid", eval_result, epoch, params.geography, params.cp_predict, params.superpop_predict)

        # every step in the scheduler is per epoch
        exp_lr_scheduler.step(eval_result.t_accr.loss_main)
        
        # logic for best model
        is_best = False
        if (epoch==start_epoch) or (eval_result.t_accr.loss_main < best_val_accr):
            best_val_accr = eval_result.t_accr.loss_main
            is_best = True
        
        if epoch!=start_epoch:
            patience = early_stopping(eval_result.t_accr.loss_main, val_prev_accr, patience, params.thresh)
            if patience == params.early_stopping_thresh:
                logging.info("Early stopping...")
                break
        
        val_prev_accr = eval_result.t_accr.loss_main

        # saving a model at every epoch
        print(f"Saving at epoch {epoch}")
        print(f'train accr: {train_result.t_accr.loss_main}, val accr: {eval_result.t_accr.loss_main}')
        checkpoint = config['model.working_dir']
        models_state_dict = [middle_models[i].state_dict() for i in range(len(middle_models))]

        save_checkpoint({
            'epoch': epoch,
            'model_state_dict': models_state_dict,
            'optimizer_state_dict': optimizer.state_dict(),
            'val_accr': eval_result._asdict(),
            'train_accr': train_result._asdict()
            }, checkpoint, is_best=is_best)
        
        
        try:
            if epoch==start_epoch: 
                params.save(''.join([config['log.dir'], 'params.json']))
                print(f"saving params at epoch:{epoch}")
        except Exception as e:
            print(f"exception while saving params:{e}")
            pass

        if epoch%20==0:
            test_result = model.valid(test_generator, plotObj=plotObj, wandb=wandb)
        plt.close('all')
        
        if wandb is not None:
            epoch_logger(wandb, "test", test_result, epoch, params.geography, params.cp_predict, params.superpop_predict)
        
        if params.hyper_search_type=='optuna':    
            trial.report(eval_result.accr.weighted_loss, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
            return best_val_accr
        
    del train_result, eval_result, model, middle_models, model_params, \
        training_generator, validation_generator
    torch.cuda.empty_cache()

if __name__=="__main__":
    config = parse_args()
    json_path = osp.join(config['data.params'], 'params.json')
    assert osp.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)
    params.dict['n_win']=0
    params.dict['chmlen']=0
    main(config, params)
