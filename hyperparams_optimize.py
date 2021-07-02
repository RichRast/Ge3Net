import os
import os.path as osp
import pickle
import optuna
import trainer
from src.utils.modelUtil import Params
from src.utils.dataUtil import set_logger
from src.main.settings_model import parse_args

logger=set_logger(__name__)

class objective(object):
    def __init__(self, params, config, save_path):
        self.params = params
        self.config = config
        self.save_path = save_path
    
    def __call__(self, trial):
        lr_aux = trial.suggest_float("lr_aux", 1e-5, 1e-1, log = True)
        self.params.learning_rate[0] = lr_aux
        
        lr_att = trial.suggest_float("lr_att", 1e-5, 1e-1, log = True)
        self.params.learning_rate[2] = lr_att
        
        lr_ffnn = trial.suggest_float("lr_ffnn", 1e-5, 1e-1, log = True)
        self.params.learning_rate[3] = lr_ffnn

        lr_lstm = trial.suggest_float("lr_lstm", 1e-5, 1e-1, log = True)
        self.params.learning_rate[4] = lr_lstm

        lr_att2 = trial.suggest_float("lr_att2", 1e-5, 1e-1, log = True)
        self.params.learning_rate[5] = lr_att2

        lr_ffnn2 = trial.suggest_float("lr_ffnn2", 1e-5, 1e-1, log = True)
        self.params.learning_rate[6] = lr_ffnn2

        lr_lstm2 = trial.suggest_float("lr_lstm2", 1e-5, 1e-1, log = True)
        self.params.learning_rate[7] = lr_lstm2

        lr_cp = trial.suggest_float("lr_cp", 1e-5, 1e-1, log = True)
        self.params.learning_rate[8] = lr_cp

        batch_size = trial.suggest_categorical("batch_size",[64, 128, 256])
        self.params.batch_size = batch_size

        # aux_net_hidden = trial.suggest_int("aux_net_hidden",100,512)
        # self.params.aux_net_hidden = aux_net_hidden
         
        aux_net_dropout = trial.suggest_float("aux_net_dropout", 0.1,0.5)
        self.params.aux_net_dropout = aux_net_dropout
        
        rnn_net_dropout = trial.suggest_float("rnn_net_dropout", 0.1,0.5)
        self.params.rnn_net_dropout = rnn_net_dropout
        
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
    save_path = osp.join(config['models.dir'], 'hyperparams_studies')
    if not osp.exists(save_path):
        os.makedirs(save_path)
    study = optuna.create_study(direction="minimize")
    study.enqueue_trial({"lr_lstm": 1e-2, "lr_lstm2": 1e-2})
    
    try:
        study.optimize(objective(params, config, save_path), n_trials=params.optuna_n_trials)
    except Exception as e:
        logger.info(f"Trial interrupted with exception: {e}, saving the study so far")
        save_study(study, save_path)

    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

    logger.info("Study statistics: ")
    logger.info(" Number of finished trials: ", len(study.trials))
    logger.info(" Number of pruned trials: ", len(pruned_trials))
    logger.info(" Number of complete trials: ", len(complete_trials))

    logger.info("Best trial: ")
    trial = study.best_trial
    
    logger.info(" Value: ", trial.value)

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
