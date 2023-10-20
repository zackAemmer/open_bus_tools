import os

import numpy as np
import torch

from obt.models import avg_speed, conv, ff, persistent, rnn, schedule, transformer
from obt.models.deeptte import DeepTTE
from obt import data_loader
from obt import data_utils


def set_feature_extraction(model, feature_extraction=True):
    if feature_extraction==False:
        for param in model.parameters():
            param.requires_grad = True
    else:
        for param in model.parameters():
            param.requires_grad = False
        # Each model must have a final named feature extraction layer
        if model.model_name=="DEEP_TTE":
            for param in model.entire_estimate.feature_extract.parameters():
                param.requires_grad = True
            for param in model.local_estimate.feature_extract.parameters():
                param.requires_grad = True
        else:
            for param in model.feature_extract.parameters():
                param.requires_grad = True

def random_param_search(hyperparameter_sample_dict, model_names):
    # Keep list of hyperparam dicts; each is randomly sampled from the given; repeat dict for each model
    set_of_random_dicts = []
    for i in range(hyperparameter_sample_dict['n_param_samples']):
        all_model_dict = {}
        random_dict = {}
        for key in list(hyperparameter_sample_dict.keys()):
            random_dict[key] = np.random.choice(hyperparameter_sample_dict[key],1)[0]
        for mname in model_names:
            all_model_dict[mname] = random_dict
        set_of_random_dicts.append(all_model_dict)
    return set_of_random_dicts

def make_one_model(model_type, hyperparameter_dict, embed_dict, config, load_weights=False, weight_folder=None, fold_num=None, skip_gtfs=False):
    # Declare base models
    base_model_list = []
    base_model_list.append(avg_speed.AvgHourlySpeedModel("AVG"))
    base_model_list.append(persistent.PersistentTimeSeqModel("PER_TIM"))
    if not skip_gtfs:
        base_model_list.append(schedule.TimeTableModel("SCH"))
    # Declare neural network models
    if model_type=="FF":
        model = ff.FF_L(
            "FF",
            n_features=8,
            hyperparameter_dict=hyperparameter_dict['FF'],
            embed_dict=embed_dict,
            collate_fn=data_loader.basic_collate_nosch,
            config=config
        )
    elif model_type=="FF_GTFS":
        model = ff.FF_L(
            "FF_GTFS",
            n_features=14,
            hyperparameter_dict=hyperparameter_dict['FF'],
            embed_dict=embed_dict,
            collate_fn=data_loader.basic_collate,
            config=config
        )
    elif model_type=="FF_GRID":
        model = ff.FF_GRID_L(
            "FF_GRID",
            n_features=14,
            n_grid_features=3*3*1,
            # n_grid_features=3*3*3*3,
            grid_compression_size=8,
            hyperparameter_dict=hyperparameter_dict['FF'],
            embed_dict=embed_dict,
            collate_fn=data_loader.basic_grid_collate,
            config=config
        )
    elif model_type=="CONV":
        model = conv.CONV_L(
            "CONV",
            n_features=4,
            hyperparameter_dict=hyperparameter_dict['CONV'],
            embed_dict=embed_dict,
            collate_fn=data_loader.sequential_collate_nosch,
            config=config
        )
    elif model_type=="CONV_GTFS":
        model = conv.CONV_L(
            "CONV_GTFS",
            n_features=10,
            hyperparameter_dict=hyperparameter_dict['CONV'],
            embed_dict=embed_dict,
            collate_fn=data_loader.sequential_collate,
            config=config
        )
    elif model_type=="CONV_GRID":
        model = conv.CONV_GRID_L(
            "CONV_GRID",
            n_features=10,
            n_grid_features=3*3*1,
            # n_grid_features=3*3*3*3,
            grid_compression_size=8,
            hyperparameter_dict=hyperparameter_dict['CONV'],
            embed_dict=embed_dict,
            collate_fn=data_loader.sequential_grid_collate,
            config=config
        )
    elif model_type=="GRU":
        model = rnn.GRU_L(
            "GRU",
            n_features=4,
            hyperparameter_dict=hyperparameter_dict['GRU'],
            embed_dict=embed_dict,
            collate_fn=data_loader.sequential_collate_nosch,
            config=config
        )
    elif model_type=="GRU_GTFS":
        model = rnn.GRU_L(
            "GRU_GTFS",
            n_features=10,
            hyperparameter_dict=hyperparameter_dict['GRU'],
            embed_dict=embed_dict,
            collate_fn=data_loader.sequential_collate,
            config=config
        )
    elif model_type=="GRU_GRID":
        model = rnn.GRU_GRID_L(
            "GRU_GRID",
            n_features=10,
            n_grid_features=3*3*1,
            # n_grid_features=3*3*3*3,
            grid_compression_size=8,
            hyperparameter_dict=hyperparameter_dict['GRU'],
            embed_dict=embed_dict,
            collate_fn=data_loader.sequential_grid_collate,
            config=config
        )
    elif model_type=="TRSF":
        model = transformer.TRSF_L(
            "TRSF",
            n_features=4,
            hyperparameter_dict=hyperparameter_dict['TRSF'],
            embed_dict=embed_dict,
            collate_fn=data_loader.sequential_collate_nosch,
            config=config
        )
    elif model_type=="TRSF_GTFS":
        model = transformer.TRSF_L(
            "TRSF_GTFS",
            n_features=10,
            hyperparameter_dict=hyperparameter_dict['TRSF'],
            embed_dict=embed_dict,
            collate_fn=data_loader.sequential_collate,
            config=config
        )
    elif model_type=="TRSF_GRID":
        model = transformer.TRSF_GRID_L(
            "TRSF_GRID",
            n_features=10,
            n_grid_features=3*3*1,
            # n_grid_features=3*3*3*3,
            grid_compression_size=8,
            hyperparameter_dict=hyperparameter_dict['TRSF'],
            embed_dict=embed_dict,
            collate_fn=data_loader.sequential_grid_collate,
            config=config
        )
    elif model_type=="DEEP_TTE":
        model = DeepTTE.Net(
            "DEEP_TTE",
            hyperparameter_dict=hyperparameter_dict['DEEPTTE'],
            collate_fn=data_loader.deeptte_collate_nosch,
            config=config
        )
    elif model_type=="DEEP_TTE_GTFS":
        model = DeepTTE.Net(
            "DEEP_TTE_GTFS",
            hyperparameter_dict=hyperparameter_dict['DEEPTTE'],
            collate_fn=data_loader.deeptte_collate,
            config=config
        )
    # Load weights if applicable
    if load_weights:
        new_base_model_list = []
        for b in base_model_list:
            new_base_model_list.append(data_utils.load_pkl(f"{weight_folder}../../../../../{b.model_name}_{fold_num}.pkl"))
            base_model_list = new_base_model_list
        last_ckpt = os.listdir(weight_folder)
        if not torch.cuda.is_available():
            model = model.load_from_checkpoint(f"{weight_folder}{last_ckpt[0]}", map_location=torch.device('cpu')).eval()
        else:
            model = model.load_from_checkpoint(f"{weight_folder}{last_ckpt[0]}").eval()
    return base_model_list, model