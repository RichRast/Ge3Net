import os
import os.path as osp
import argparse
import sys
import pickle
import copy

import torch.nn as nn
import torch.nn.functional as F

import optuna
import wandb
import trainer
from settings import parse_args

class objective(object):
    def __init__(self, params, root_model_dir):
        self.params = params
        self.config = config
    
    def __call__(self, trial):
        lr0 = trial.suggest_float("lr0", 1e-5, 1e-1, log = True)
        self.params.learning_rate[0] = lr0
        
        lr1 = trial.suggest_float("lr1", 1e-5, 1e-1, log = True)
        self.params.learning_rate[1] = lr1
        
        aux_hidden = trial.suggest_int("num_hidden",100,512)
        self.params.aux_net_hidden = aux_hidden
        
        lr2 = trial.suggest_float("lr2", 1e-5, 1e-1, log = True)
        self.params.learning_rate[2] = lr2
        
        #rnn_hidden_unit = trial.suggest_int("rnn_hidden_unit",32,128)
        #self.params.rnn_net["hidden_size"] = rnn_hidden_unit
        tbptt = trial.suggest_int("tbptt", 0,1)
        self.params.tbptt = tbptt
        
        aux_dropout = trial.suggest_float("aux_dropout", 0.1,0.5)
        self.params.aux_net_dropout = aux_dropout
        
        rnn_dropout = trial.suggest_float("rnn_dropout", 0.1,0.5)
        self.params.rnn_net_dropout = rnn_dropout
        
        # save the params in json file
        json_path = osp.join(config['model.working_dir'], config['model_version'])
        if not osp.exists(json_path):
            os.mkdir(json_path)
        params_filename = ''.join(['_trial_', str(trial.number), 'params.json'])
        params.save(osp.join(json_path, params_filename))
        
        accuracy = trainer.main(self.config, self.params, trial)
        return accuracy
    
    
def save_study(study, save_filename):
    if not osp.exists(save_filename):
        os.mkdir(save_filename)
        
    # save the study
    print("Saving the study")
    with open(osp.join(save_filename, "study.pkl"), 'wb') as f:
        pickle.dump(study, f)
        
class wandb_trainer():
    def __init__(self, config, params):
        self.params = params
        self.config = config
    
    def __call__(self):
        trainer.main(self.config, self.params)
        
def main(config, params):
    
    save_filename = osp.join(config['model.working_dir'], config['model_version'])
    if params.hyper_search_type=="optuna":       
        study = optuna.create_study(direction="minimize", study_name=''.join(["study", config['model_version']]))
        study.enqueue_trial({"lr1": 1e-2})
        
        try:
            study.optimize(objective(params, config), n_trials=params.optuna_n_trials)
        except:
            print("Trial interrupted, saving the study so far")
            save_study(study, save_filename)

        pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
        complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

        print("Study statistics: ")
        print(" Number of finished trials: ", len(study.trials))
        print(" Number of pruned trials: ", len(pruned_trials))
        print(" Number of complete trials: ", len(complete_trials))

        print("Best trial: ")
        trial = study.best_trial
        
        print(" Value: ", trial.value)

        print(" Params: ")
        for key, value in trial.params.items():
            print(" {}: {}".format(key, value))
            
        # save the sudy if all trials got completed
        save_study(study, save_filename)
                
    elif params.hyper_search_type=="wandb":
        sweep_config = params.wandb_config
        job_name = "wandb_sweep"
        sweep_id = wandb.sweep(sweep_config, project=job_name)
        if len(params.sweep_id)==8:
            sweep_id = params.sweep_id
        wb_train = wandb_trainer(config, params)
        wandb.agent(sweep_id, function = wb_train, count=params.wandb_n_trials)
        
    else:
        # optimize learning rate for group 0
        learning_rates=[1e-6, 1e-2, 1e-4]

        for lr in learning_rates:
            params.learning_rate[0] = lr

            # launch the training job
            job_name = "learning_rate_{}".format(lr)
            
            # write the parameters to a json file
            json_path = osp.join(save_filename, ''.join(['lr_', str(lr), 'params.json']))
            params.save(json_path)
    
            accuracy = trainer.main(config, params)
            print(f'accuracy:{accuracy}')

if __name__=="__main__":
    config, params = parse_args()
    config['model_version'] = ''.join([str(params.model), '_', str(params.major_version), '.', str(params.minor_version)])
    main(config, params)
