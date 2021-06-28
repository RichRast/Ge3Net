import torch
import numpy as np
import pandas as pd
import os.path as osp
import logging
from src.utils.modelUtil import  Params, weight_int
from src.utils.dataUtil import set_logger, load_path
from src.utils.labelUtil import repeat_pop_arr
from src.utils.decorators import timer
from src.main.dataset import Haplotype
from src.models.modelSelection import modelSelect
from src.models.modelParamsSelection import Selections
from src.main.settings_model import parse_args
from src.main.visualization import Plot_per_epoch
from src.models.Ge3Net import Ge3NetBase

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
    
    labels_path = config['data.labels']
    data_path=config['data.dir']
    plotObj=None
    if config['log.verbose']:
        # Set the logger
        logger=set_logger(config['models.dir'])
        wandb.init(project='Ge3Net', config=params, allow_val_change=True)
        wandb.run.name='_'.join([str(params.model), str(config['model.summary'])])
        # params=wandb.config

        # Initiate the class for plotting per epoch
        if params.plotting and labels_path is not None:
            pop_dict = load_path(osp.join(labels_path, 'granular_pop.pkl'), en_pickle=True)
            rev_pop_dict = {v:k for k,v in pop_dict.items()}
            pop_sample_map = pd.read_csv(osp.join(labels_path, params.pop_sample_map), sep='\t')
            pop_arr = repeat_pop_arr(pop_sample_map)
            plotObj = Plot_per_epoch(params.n_comp_overall, params.n_comp_subclass, \
            rev_pop_dict, pop_arr, pop_order=params.pop_num, geography=params.geography)
        
    # configure device
    params.device = torch.device(config['cuda'] if params.cuda else 'cpu')

    #============================= Create and load datasets ===============================#
    # Create the input data pipeline
    logger.info("Loading the datasets...")
    training_dataset = Haplotype('train', params, data_path, labels_path=labels_path)
    validation_dataset = Haplotype('valid', params, data_path, labels_path=labels_path)
    training_generator = torch.utils.data.DataLoader(training_dataset, batch_size=params.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    validation_generator = torch.utils.data.DataLoader(validation_dataset, batch_size=params.batch_size, num_workers=0, pin_memory=True)
    test_generator=None
    if params.evaluateTest:
        test_dataset = Haplotype('test', params, data_path, labels_path=labels_path)
        test_generator = torch.utils.data.DataLoader(test_dataset, batch_size=params.batch_size, num_workers=0, pin_memory=True)
         
    #============================= Create and load the model ===============================#    
    modelOption=modelSelect.get_selection()
    option = Selections.get_selection()
    criterion = option['loss'][params.criteria](reduction='sum', alpha=params.criteria_alpha, geography=params.geography)
    cp_criterion=option['cpMetrics']['loss_cp']
    model = modelOption['models'][params.model](params, criterion, cp_criterion)
    model.to(params.device)
    model.apply(weight_int)

    logger.info(f"is the model on cuda? : {next(model.parameters()).is_cuda}")
    if torch.cuda.device_count() > 1:
        logger.info("Using", torch.cuda.device_count(), "GPUs")
        model = torch.nn.DataParallel(model, dim=0)

    if config['log.verbose']:
        #watch all the models
        wandb.log({"model_name":str(model)})
        wandb.watch(model, log='all')
 
    Ge3NetTrainer=Ge3NetBase(params, model, option)
    Ge3NetTrainer.launchTraining(config['models.dir'], training_generator, validation_generator, test_generator=test_generator, \
    plotObj=plotObj, wandb=(lambda x: wandb if x else None)(config['log.verbose']))

if __name__=="__main__":
    config = parse_args()
    json_path = osp.join(config['data.params'], 'params.json')
    assert osp.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)
    params.dict['n_win']=0 # these are set during data load
    params.dict['chmlen']=0
    main(config, params)
