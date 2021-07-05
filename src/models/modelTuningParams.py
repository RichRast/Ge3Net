
suggestParamsDict= {
                    "Model_O":{
                                "lr_dict":{
                                    "lr_aux": (1e-5, 1e-1), "lr_att": (1e-5, 1e-1), "lr_ffnn": (1e-5, 1e-1),
                                    "lr_lstm": (1e-5, 1e-1), "lr_att2": (1e-5, 1e-1), "lr_ffnn2": (1e-5, 1e-1),
                                    "lr_lstm2": (1e-5, 1e-1), "lr_cp": (1e-5, 1e-1)
                                            },
                                
                                "int": {
                                        "cp_tol": (1,3),
                                        },
                                "categorical": {
                                        "aux_net_hidden": [64, 100, 128, 256],
                                        "aux_net_hidden1": [64, 100, 128, 256],
                                        "aux_net_hidden2": [50, 64, 100],
                                        "aux_next_norm": ["LayerNorm", "BatchNorm"],
                                        "FFNN_output":[64, 100, 128],
                                        "rnn_net_out1":[64, 100, 128],
                                        "rnn_net_hidden": [16, 32, 64],
                                        "batch_size": [64, 128, 256],
                                        "win_size":[100,500,1000,1500, 2000]   
                                },
                                "float": {
                                        "aux_net_dropout": (0.1,0.5),
                                        "aux_net_dropout1": (0.1,0.5),
                                        "aux_net_dropout2": (0.1,0.5),
                                        "att_dropout": (0.1,0.5),
                                        "att_dropout2": (0.1,0.5),
                                        "rnn_net_dropout": (0.1,0.5),
                                        "logits_Block_dropout": (0.1,0.5),
                                        "FFNN_dropout1":(0.1,0.5),
                                        "FFNN_dropout2":(0.1,0.5),
                                        }
                                }, 
                    
                    "Model_N":{
                                "lr_dict":{
                                    "lr_aux": (1e-5, 1e-1), "lr_att": (1e-5, 1e-1), "lr_ffnn": (1e-5, 1e-1),
                                    "lr_lstm": (1e-5, 1e-1), "lr_lstm2": (1e-5, 1e-1), "lr_cp": (1e-5, 1e-1)
                                            },
                                
                                "int": {
                                        "cp_tol": (1,3),
                                        },
                                "categorical": {
                                        "aux_net_hidden": [64, 100, 128, 256],
                                        "aux_net_hidden1": [64, 100, 128, 256],
                                        "aux_net_hidden2": [50, 64, 100],
                                        "aux_next_norm": ["LayerNorm", "BatchNorm"],
                                        "FFNN_output":[64, 100, 128],
                                        "rnn_net_out1":[64, 100, 128],
                                        "rnn_net_hidden": [16, 32, 64],
                                        "batch_size": [64, 128, 256],
                                        "win_size":[100,500,1000,1500, 2000]   
                                },
                                "float": {
                                        "aux_net_dropout": (0.1,0.5),
                                        "aux_net_dropout1": (0.1,0.5),
                                        "aux_net_dropout2": (0.1,0.5),
                                        "att_dropout": (0.1,0.5),
                                        "att_dropout2": (0.1,0.5),
                                        "rnn_net_dropout": (0.1,0.5),
                                        "logits_Block_dropout": (0.1,0.5),
                                        "FFNN_dropout1":(0.1,0.5),
                                        "FFNN_dropout2":(0.1,0.5),
                                        }
                                },

                    "Model_M":{
                                "lr_dict":{
                                    "lr_aux": (1e-5, 1e-1), "lr_mlp": (1e-5, 1e-1),"lr_cp": (1e-5, 1e-1)
                                            },
                                
                                "int": {
                                        "cp_tol": (1,3),
                                        },
                                "categorical": {
                                        "aux_net_hidden": [64, 100, 128, 256],
                                        "aux_net_hidden1": [64, 100, 128, 256],
                                        "aux_net_hidden2": [50, 64, 100],
                                        "aux_next_norm": ["LayerNorm", "BatchNorm"],
                                        "mlp_net_hidden":[64, 100, 128, 256],
                                        "batch_size": [64, 128, 256],
                                        "win_size":[100,500,1000,1500, 2000]   
                                },
                                "float": {
                                        "aux_net_dropout": (0.1,0.5),
                                        "aux_net_dropout1": (0.1,0.5),
                                        "aux_net_dropout2": (0.1,0.5),
                                        "mlp_dropout": (0.1,0.5),
                                        "logits_Block_dropout": (0.1,0.5),
                                        }
                                },  
                    
                    "Model_L":{
                                "lr_dict":{
                                    "lr_base": (1e-5, 1e-1), "lr_attBlock": (1e-5, 1e-1), "lr_lstm":(1e-5, 1e-1),
                                    "lr_cp": (1e-5, 1e-1)
                                            },
                                
                                "int": {
                                        "cp_tol": (1,3),
                                        },
                                "categorical": {
                                        "aux_net_hidden": [64, 100, 128, 256],
                                        "aux_net_hidden1": [64, 100, 128, 256],
                                        "aux_net_hidden2": [50, 64, 100],
                                        "aux_next_norm": ["LayerNorm", "BatchNorm"],
                                        "FFNN_output":[64, 100, 128, 256],
                                        "rnn_net_out":[64, 100, 128, 256],
                                        "rnn_net_hidden": [16, 32, 64, 128],
                                        "batch_size": [64, 128, 256],
                                        "win_size":[100,500,1000,1500, 2000]   
                                },
                                "float": {
                                        "aux_net_dropout": (0.1,0.5),
                                        "aux_net_dropout1": (0.1,0.5),
                                        "aux_net_dropout2": (0.1,0.5),
                                        "att_dropout": (0.1,0.5),
                                        "att_dropout2": (0.1,0.5),
                                        "rnn_net_dropout": (0.1,0.5),
                                        "logits_Block_dropout": (0.1,0.5),
                                        "FFNN_dropout1":(0.1,0.5),
                                        "FFNN_dropout2":(0.1,0.5),
                                        }
                                },

                    "Model_H":{
                                "lr_dict":{
                                    "lr_aux": (1e-5, 1e-1), "lr_att": (1e-5, 1e-1), "lr_ffnn": (1e-5, 1e-1), 
                                    "lr_lstm":(1e-5, 1e-1),
                                    "lr_cp": (1e-5, 1e-1)
                                            },
                                
                                "int": {
                                        "cp_tol": (1,3),
                                        },
                                "categorical": {
                                        "aux_net_hidden": [64, 100, 128, 256],
                                        "aux_net_hidden1": [64, 100, 128, 256],
                                        "aux_net_hidden2": [50, 64, 100],
                                        "aux_next_norm": ["LayerNorm", "BatchNorm"],
                                        "FFNN_output":[64, 100, 128],
                                        "rnn_net_hidden": [16, 32, 64],
                                        "batch_size": [64, 128, 256],
                                        "win_size":[100,500,1000,1500, 2000]   
                                },
                                "float": {
                                        "aux_net_dropout": (0.1,0.5),
                                        "aux_net_dropout1": (0.1,0.5),
                                        "aux_net_dropout2": (0.1,0.5),
                                        "att_dropout": (0.1,0.5),
                                        "att_dropout2": (0.1,0.5),
                                        "rnn_net_dropout": (0.1,0.5),
                                        "logits_Block_dropout": (0.1,0.5),
                                        "FFNN_dropout1":(0.1,0.5),
                                        "FFNN_dropout2":(0.1,0.5),
                                        }
                                },
                    
                    "Model_F":{
                                "lr_dict":{
                                    "lr_aux": (1e-5, 1e-1), "lr_att": (1e-5, 1e-1), "lr_ffnn": (1e-5, 1e-1), 
                                    "lr_cp": (1e-5, 1e-1)
                                            },
                                
                                "int": {
                                        "cp_tol": (1,3),
                                        },
                                "categorical": {
                                        "aux_net_hidden": [64, 100, 128, 256],
                                        "aux_net_hidden1": [64, 100, 128, 256],
                                        "aux_net_hidden2": [50, 64, 100],
                                        "aux_next_norm": ["LayerNorm", "BatchNorm"],
                                        "batch_size": [64, 128, 256],
                                        "win_size":[100,500,1000,1500, 2000]   
                                },
                                "float": {
                                        "aux_net_dropout": (0.1,0.5),
                                        "aux_net_dropout1": (0.1,0.5),
                                        "aux_net_dropout2": (0.1,0.5),
                                        "att_dropout": (0.1,0.5),
                                        "att_dropout2": (0.1,0.5),
                                        "logits_Block_dropout": (0.1,0.5),
                                        "FFNN_dropout1":(0.1,0.5),
                                        "FFNN_dropout2":(0.1,0.5),
                                        }
                                },

                    "Model_D":{
                                "lr_dict":{
                                    "lr_aux": (1e-5, 1e-1),
                                    "lr_lstm":(1e-5, 1e-1),
                                    "lr_cp": (1e-5, 1e-1)
                                            },
                                
                                "int": {
                                        "cp_tol": (1,3),
                                        },
                                "categorical": {
                                        "aux_net_hidden": [64, 100, 128, 256],
                                        "aux_net_hidden1": [64, 100, 128, 256],
                                        "aux_net_hidden2": [50, 64, 100],
                                        "aux_next_norm": ["LayerNorm", "BatchNorm"],
                                        "rnn_net_hidden": [16, 32, 64, 128],
                                        "batch_size": [64, 128, 256],
                                        "win_size":[100,500,1000,1500, 2000]   
                                },
                                "float": {
                                        "aux_net_dropout": (0.1,0.5),
                                        "aux_net_dropout1": (0.1,0.5),
                                        "aux_net_dropout2": (0.1,0.5),
                                        "rnn_net_dropout": (0.1,0.5),
                                        "logits_Block_dropout": (0.1,0.5),
                                        }
                                },
                    }   
 
def getParamKeys(params):
    params_dict = suggestParamsDict[params.model]
    paramsKeys=list(['learning_rate'])
    paramsKeys.extend(list(params_dict["categorical"].keys()))
    paramsKeys.extend(list(params_dict["int"].keys())) 
    paramsKeys.extend(list(params_dict["float"].keys()))
    return paramsKeys

def suggestParams(params, trial):
    suggestedParams={}
    if params.model == "Model_O":
        lr_dict = suggestParamsDict[params.model]["lr_dict"]
        params.learning_rate[0] = trial.suggest_float("lr_aux", lr_dict["lr_aux"][0], lr_dict["lr_aux"][1], log=True)
        params.learning_rate[2] = trial.suggest_float("lr_att", lr_dict["lr_att"][0], lr_dict["lr_att"][1], log=True)
        params.learning_rate[3] = trial.suggest_float("lr_ffnn", lr_dict["lr_ffnn"][0], lr_dict["lr_ffnn"][1], log=True)
        params.learning_rate[4] = trial.suggest_float("lr_lstm", lr_dict["lr_lstm"][0], lr_dict["lr_lstm"][1], log=True)
        params.learning_rate[5] = trial.suggest_float("lr_att2", lr_dict["lr_att2"][0], lr_dict["lr_att2"][1], log=True)
        params.learning_rate[6] = trial.suggest_float("lr_ffnn2", lr_dict["lr_ffnn2"][0], lr_dict["lr_ffnn2"][1], log=True)
        params.learning_rate[7] = trial.suggest_float("lr_lstm2", lr_dict["lr_lstm2"][0], lr_dict["lr_lstm2"][1], log=True)
        params.learning_rate[8] = trial.suggest_float("lr_cp", lr_dict["lr_cp"][0], lr_dict["lr_cp"][1], log=True)
    elif params.model == "Model_N":
        lr_dict = suggestParamsDict[params.model]["lr_dict"]
        params.learning_rate[0] = trial.suggest_float("lr_aux", lr_dict["lr_aux"][0], lr_dict["lr_aux"][1], log=True)
        params.learning_rate[1] = trial.suggest_float("lr_lstm", lr_dict["lr_lstm"][0], lr_dict["lr_lstm"][1], log=True)
        params.learning_rate[3] = trial.suggest_float("lr_att", lr_dict["lr_att"][0], lr_dict["lr_att"][1], log=True)
        params.learning_rate[4] = trial.suggest_float("lr_ffnn", lr_dict["lr_ffnn"][0], lr_dict["lr_ffnn"][1], log=True)
        params.learning_rate[5] = trial.suggest_float("lr_lstm2", lr_dict["lr_lstm2"][0], lr_dict["lr_lstm2"][1], log=True)
        params.learning_rate[6] = trial.suggest_float("lr_cp", lr_dict["lr_cp"][0], lr_dict["lr_cp"][1], log=True)
    elif params.model == "Model_M":
        lr_dict = suggestParamsDict[params.model]["lr_dict"]
        params.learning_rate[0] = trial.suggest_float("lr_aux", lr_dict["lr_aux"][0], lr_dict["lr_aux"][1], log=True)
        params.learning_rate[1] = trial.suggest_float("lr_mlp", lr_dict["lr_mlp"][0], lr_dict["lr_mlp"][1], log=True)
        params.learning_rate[2] = trial.suggest_float("lr_cp", lr_dict["lr_cp"][0], lr_dict["lr_cp"][1], log=True)
    elif params.model == "Model_L":
        lr_dict = suggestParamsDict[params.model]["lr_dict"]
        params.learning_rate[0] = trial.suggest_float("lr_base", lr_dict["lr_base"][0], lr_dict["lr_base"][1], log=True)
        params.learning_rate[2] = trial.suggest_float("lr_attBlock", lr_dict["lr_attBlock"][0], lr_dict["lr_attBlock"][1], log=True)
        params.learning_rate[3] = trial.suggest_float("lr_lstm", lr_dict["lr_lstm"][0], lr_dict["lr_lstm"][1], log=True)
        params.learning_rate[4] = trial.suggest_float("lr_cp", lr_dict["lr_cp"][0], lr_dict["lr_cp"][1], log=True)
    elif params.model == "Model_H":
        lr_dict = suggestParamsDict[params.model]["lr_dict"]
        params.learning_rate[0] = trial.suggest_float("lr_aux", lr_dict["lr_aux"][0], lr_dict["lr_aux"][1], log=True)
        params.learning_rate[2] = trial.suggest_float("lr_att", lr_dict["lr_att"][0], lr_dict["lr_att"][1], log=True)
        params.learning_rate[3] = trial.suggest_float("lr_ffnn", lr_dict["lr_ffnn"][0], lr_dict["lr_ffnn"][1], log=True)
        params.learning_rate[4] = trial.suggest_float("lr_lstm", lr_dict["lr_lstm"][0], lr_dict["lr_lstm"][1], log=True)
        params.learning_rate[5] = trial.suggest_float("lr_cp", lr_dict["lr_cp"][0], lr_dict["lr_cp"][1], log=True)
    elif params.model == "Model_F":
        lr_dict = suggestParamsDict[params.model]["lr_dict"]
        params.learning_rate[0] = trial.suggest_float("lr_aux", lr_dict["lr_aux"][0], lr_dict["lr_aux"][1], log=True)
        params.learning_rate[2] = trial.suggest_float("lr_att", lr_dict["lr_att"][0], lr_dict["lr_att"][1], log=True)
        params.learning_rate[3] = trial.suggest_float("lr_ffnn", lr_dict["lr_ffnn"][0], lr_dict["lr_ffnn"][1], log=True)
        params.learning_rate[4] = trial.suggest_float("lr_cp", lr_dict["lr_cp"][0], lr_dict["lr_cp"][1], log=True)
    elif params.model == "Model_D":
        lr_dict = suggestParamsDict[params.model]["lr_dict"]
        params.learning_rate[0] = trial.suggest_float("lr_aux", lr_dict["lr_aux"][0], lr_dict["lr_aux"][1], log=True)
        params.learning_rate[1] = trial.suggest_float("lr_lstm", lr_dict["lr_lstm"][0], lr_dict["lr_lstm"][1], log=True)
        params.learning_rate[2] = trial.suggest_float("lr_cp", lr_dict["lr_cp"][0], lr_dict["lr_cp"][1], log=True)

    for param in suggestParamsDict[params.model]["categorical"].keys():
        suggestedParams[param] = trial.suggest_categorical(param, \
                                suggestParamsDict[params.model]["categorical"][param])

    for param in suggestParamsDict[params.model]["int"].keys():
        suggestedParams[param] = trial.suggest_int(param, suggestParamsDict[params.model]["int"][param][0], 
                                suggestParamsDict[params.model]["int"][param][1])

    for param in suggestParamsDict[params.model]["float"].keys():
        suggestedParams[param] = trial.suggest_float(param, suggestParamsDict[params.model]["float"][param][0],
                                suggestParamsDict[params.model]["float"][param][1])

    params.update(suggestedParams)
    return params