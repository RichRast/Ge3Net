import os.path as osp
import torch
import numpy as np
from src.models import Model_A, Model_B, Model_C
from src.models.modelSelection import modelSelect
from src.models.modelParamsSelection import Selections
from src.utils.modelUtil import load_model, Params
from src.utils.decorators import timer
from src.main.dataset import Haplotype
from src.main.settings_model import parse_args
from src.models.Ge3Net import Ge3NetBase

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
          
    # configure device
    params.device = torch.device(config['cuda'] if params.cuda else 'cpu')
    print(f" device used: {params.device}")

    # Create the input data pipeline
    print("Loading the datasets...")

    labels_path = config['data.labels']
    data_path=config['data.dir']
    test_dataset = Haplotype(config['data.dataset_type'], params, data_path, labels_path=labels_path)
    test_generator = torch.utils.data.DataLoader(test_dataset, batch_size=params.batch_size, num_workers=0, pin_memory=True)
         
    #================================ Create the model ================================
    model_path = osp.join(config['models.dir'], 'models_dir')
    modelOption=modelSelect.get_selection()
    option = Selections.get_selection()
    criterion = option['loss'][params.criteria](reduction='sum', alpha=params.criteria_alpha, geography=params.geography)
    cp_criterion=option['cpMetrics']['loss_cp']
    model_init = modelOption['models'][params.model](params, criterion, cp_criterion)
    if config['model.loadBest']:
        model = load_model(''.join([str(model_path),'/best.pt']), model_init)
    else:
        model= load_model(''.join([str(model_path),'/last.pt']), model_init)
    model.to(params.device)
    model.eval()
    print(f"is the model on cuda? : {next(model.parameters()).is_cuda}")

    #================================ Create the model ================================
    Ge3NetTrainer=Ge3NetBase(params, model, option)
    if labels_path is not None:
        test_result = Ge3NetTrainer.batchLoopValid(test_generator)
    else:
         test_result = Ge3NetTrainer.pred(test_generator)
    
    return test_result, test_dataset, model
    
if __name__=="__main__":
    config = parse_args()
    json_path = osp.join(config['models.dir'], 'params.json')
    assert osp.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)
    params.rtnOuts=True
    main(config, params)
