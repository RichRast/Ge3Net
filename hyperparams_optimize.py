import os
import os.path as osp
import pickle
import optuna
import trainer
from src.utils.modelUtil import Params
from src.utils.dataUtil import set_logger
from src.main.settings_model import parse_args
from src.models.modelTuningParams import suggestParams, getParamKeys

logger=set_logger(__name__)

class objective(object):
    def __init__(self, params, config, save_path):
        self.params = params
        self.config = config
        self.save_path = save_path
    
    def __call__(self, trial):
        self.params = suggestParams(self.params, trial)
        
        # save the params in yaml file
        params_filename = ''.join(['_trial_', str(trial.number), 'params.yaml'])
        params.save(osp.join(self.save_path, params_filename))
        
        accuracy = trainer.main(self.config, self.params, trial=trial)
        
        return accuracy
    
def save_study(study, save_path):    
    # save the study
    logger.info("Saving the study")
    with open(osp.join(save_path, "study.pkl"), 'wb') as f:
        pickle.dump(study, f)
                
def main(config, params):
    params.optuna_n_trials=10
    params.evalBalancedGcd=False
    params.evaluateTest=False
    save_path = osp.join(config['models.dir'], 'hyperparams_studies')
    if not osp.exists(save_path):
        os.makedirs(save_path)
    study = optuna.create_study(direction="minimize", pruner=optuna.pruners.HyperbandPruner())
    init_params=getParamKeys(params)
    study.enqueue_trial({k:params[k] for k in init_params})
    
    try:
        study.optimize(objective(params, config, save_path), n_trials=params.optuna_n_trials)
    except Exception as e:
        logger.info(f"Trial interrupted with exception: {e}, saving the study so far")
        save_study(study, save_path)

    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

    logger.info("Study statistics: ")
    logger.info(f" Number of pruned trials: , {len(pruned_trials)}")
    logger.info(f" Number of complete trials:, {len(complete_trials)}")

    logger.info("Best trial: ")
    trial = study.best_trial
    
    logger.info(f" Value:, {trial.value}")

    logger.info(" Params: ")
    for key, value in trial.params.items():
        logger.info(" {}: {}".format(key, value))
        
    # save the sudy if all trials got completed
    save_study(study, save_path)
                
if __name__=="__main__":
    config = parse_args()
    yaml_path = osp.join(config['data.params'], 'params.yaml')
    assert osp.isfile(yaml_path), "No yaml configuration file found at {}".format(yaml_path)
    params = Params(yaml_path)
    main(config, params)
