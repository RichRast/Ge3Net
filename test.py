import os.path as osp
import torch
import numpy as np
from src.models import LSTM, AuxiliaryTask, Conv, Attention, Transformer, BasicBlock, Model_A, Model_B, Model_C
from src.utils.modelUtil import load_model, Params
from src.utils.decorators import timer
from src.main.dataset import Haplotype
from src.main.settings_model import parse_args, MODEL_CLASS

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
    model_subclass, model_basics = MODEL_CLASS[params.model]
    middle_models=[]
    
    for i, model_basic in enumerate(model_basics):
        # instantiate the model class
        m = eval(model_basic)(params).to(params.device)
        middle_models.append(m)
        print(f'model {model_subclass} : {model_basic}')
            
    model_path = osp.join(config['models.dir'], 'models_dir')
    if config['model.loadBest']:
        model_ret = load_model(''.join([str(model_path),'/best.pt']), middle_models)
    else:
        model_ret = load_model(''.join([str(model_path),'/last.pt']), middle_models)
    # call to the corresponding model, example - Model_L.model_L
    model = eval(model_subclass[0])(*model_ret, params=params)
    #================================ Create the model ================================
    
    if labels_path is not None:
        test_result = model.valid(test_generator)
    else:
        test_result = model.pred(test_generator)
    
    return test_result, test_dataset
    
if __name__=="__main__":
    config = parse_args()
    json_path = osp.join(config['models.dir'], 'params.json')
    assert osp.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)
    params.rtnOuts=True
    main(config, params)
