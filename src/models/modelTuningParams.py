
lr_dict_model_O = {"lr_aux": (1e-5, 1e-1), "lr_att": (1e-5, 1e-1)}

params_dict_model_O = {"int": {"aux_net_hidden1": (64,512)},
                       "categorical": {"aux_net_hidden": [64, 100, 128, 256]},
                       "float": {"aux_net_dropout": (0.1,0.5)} }
def getParamKeys(params):
    if params.model == "Model_O":
        params_dict = params_dict_model_O
    
    paramsKeys=list(['learning_rate'])
    paramsKeys.extend(list(params_dict["categorical"].keys()))
    paramsKeys.extend(list(params_dict["int"].keys())) 
    paramsKeys.extend(params_dict["float"].keys())
    return paramsKeys

def suggestParams(params, trial):
    suggestedParams={}
    if params.model == "Model_O":
        lr_dict = lr_dict_model_O
        params_dict = params_dict_model_O
        params.learning_rate[0] = trial.suggest_float("lr_aux", lr_dict["lr_aux"][0],
                                                        lr_dict["lr_aux"][1])

    for param in params_dict["categorical"].keys():
        suggestedParams[param] = trial.suggest_categorical(param, params_dict["categorical"][param])

    for param in params_dict["int"].keys():
        suggestedParams[param] = trial.suggest_int(param, params_dict["int"][param][0], 
                                                    params_dict["int"][param][1])

    for param in params_dict["float"].keys():
        suggestedParams[param] = trial.suggest_float(param, params_dict["float"][param][0],
                                                     params_dict["float"][param][1])

    params.update(suggestedParams)
    return params

    # suggestedLr={
    #     "lr_aux": trial.suggest_float("lr_aux", 1e-5, 1e-1, log = True),  
    #     # "lr_att":lambda: trial.suggest_float("lr_att", 1e-5, 1e-1, log = True),
    #     # "lr_ffnn":lambda: trial.suggest_float("lr_ffnn", 1e-5, 1e-1, log = True),
    #     # "lr_lstm":lambda: trial.suggest_float("lr_lstm", 1e-5, 1e-1, log = True),
    #     # "lr_att2":lambda: trial.suggest_float("lr_att2", 1e-5, 1e-1, log = True),
    #     # "lr_ffnn2":lambda: trial.suggest_float("lr_ffnn2", 1e-5, 1e-1, log = True),
    #     # "lr_lstm2":lambda: trial.suggest_float("lr_lstm2", 1e-5, 1e-1, log = True),
    #     # "lr_cp":lambda: trial.suggest_float("lr_cp", 1e-5, 1e-1, log = True),
    # }
    # params.learning_rate[0]=suggestedLr["lr_aux"]
    # print(f"suggestedLr:{suggestedLr}")
    # print(f"params.learning_rate[0]:{params.learning_rate[0]}")
    # # params.learning_rate[2]=suggestedLr["lr_att"]
    # # params.learning_rate[3]=suggestedLr["lr_ffnn"]
    # # params.learning_rate[4]=suggestedLr["lr_lstm"]
    # # params.learning_rate[5]=suggestedLr["lr_att2"]
    # # params.learning_rate[6]=suggestedLr["lr_ffnn2"]
    # # params.learning_rate[7]=suggestedLr["lr_lstm2"]
    # # params.learning_rate[8]=suggestedLr["lr_cp"]


    # suggestedParams={
    #     # "batch_size":trial.suggest_categorical("batch_size", [64, 128, 256]),
    #     "aux_net_hidden": trial.suggest_int("aux_net_hidden", 100,512),
    #     # "aux_net_hidden1":lambda: trial.suggest_int("aux_net_hidden1", 64,512),
    #     # "aux_net_hidden2":lambda: trial.suggest_int("aux_net_hidden2",64,512),   
    #     # "aux_net_dropout":lambda: trial.suggest_float("aux_net_dropout", 0.1,0.5),
    #     # "aux_net_dropout1":lambda: trial.suggest_float("aux_net_dropout1", 0.1,0.5),
    #     # "aux_net_dropout2":lambda: trial.suggest_float("aux_net_dropout2", 0.1,0.5),
    #     # "aux_next_norm":lambda: trial.suggest_categorical("aux_next_norm", ["LayerNorm", "BatchNorm"]),
    #     # "rnn_net_dropout":lambda: trial.suggest_float("rnn_net_dropout", 0.1,0.5),
    # }
    
    
    # #newParams={k:suggestedParams.get(k) for k in suggestedParams.keys()}
    # #params.update(newParams)
    # params.update(suggestedParams)
    # return params
    