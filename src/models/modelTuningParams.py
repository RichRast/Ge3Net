
suggestParamsDict= {
                    "Model_O":{
                                "lr_dict":{
                                    "lr_aux": (1e-5, 1e-1), "lr_att": (1e-5, 1e-1), "lr_ffnn": (1e-5, 1e-1),
                                    "lr_lstm": (1e-5, 1e-1), "lr_att2": (1e-5, 1e-1), "lr_ffnn2": (1e-5, 1e-1),
                                    "lr_lstm2": (1e-5, 1e-1), "lr_cp": (1e-5, 1e-1)
                                            },
                                
                                "int": {
                                        "win_size":(500,1200) ,
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
                                       "win_size":(500,1200) ,
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
                                        "win_size":(500,1200) ,
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
                                        "win_size":(500,1200) ,
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
                                        "win_size":(500,1200),
                                        # "train_seed": (42,1235) 
                                        },
                                "categorical": {
                                        "aux_net_hidden": [64, 100, 128, 256],
                                        "aux_net_hidden1": [64, 100, 128, 256],
                                        "aux_net_hidden2": [50, 64, 100],
                                        "aux_next_norm": ["LayerNorm", "BatchNorm"],
                                        "FFNN_output":[64, 103, 128],
                                        "rnn_net_hidden": [16, 32, 64],
                                        "batch_size": [64, 128, 256],  
                                        
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
                                        "win_size":(500,1200) ,
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
                                        "win_size":(500,1200) ,
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
                "Model_Q":{

                                "categorical": {
                                        # "mht_hidden_dim": [64, 100, 128, 256, 512, 2048],
                                        "mht_num_heads":(1,2,4),
                                        "mht_nlayers":(1,2,4),
                                },
                                },
                    }   
typeKeys=['categorical', 'int', 'float']
def getParamKeys(params):
    params_dict = suggestParamsDict[params.model]
    paramsKeys=list(['learning_rate'])
    
    for k in typeKeys:
        if params_dict.get(k) is not None:
            paramsKeys.extend(list(params_dict[k].keys()))
    return paramsKeys

def suggestParams(params, trial):
    suggestedParams={}
#     lr_dict = suggestParamsDict[params.model]["lr_dict"]
#     i=0
#     for k, v in lr_dict.items():
#         params.learning_rate[i] = trial.suggest_float(k, v[0], v[1], log=True)
#         i+=1
    if suggestParamsDict[params.model].get('categorical') is not None: 
        for param in suggestParamsDict[params.model]["categorical"].keys():
                suggestedParams[param] = trial.suggest_categorical(param, \
                                        suggestParamsDict[params.model]["categorical"][param])
    if suggestParamsDict[params.model].get('int') is not None: 
        for param in suggestParamsDict[params.model]["int"].keys():
                suggestedParams[param] = trial.suggest_int(param, suggestParamsDict[params.model]["int"][param][0], 
                                        suggestParamsDict[params.model]["int"][param][1])

    if suggestParamsDict[params.model].get('float') is not None:
        for param in suggestParamsDict[params.model]["float"].keys():
                suggestedParams[param] = trial.suggest_float(param, suggestParamsDict[params.model]["float"][param][0],
                                suggestParamsDict[params.model]["float"][param][1])

    params.update(suggestedParams)
    return params