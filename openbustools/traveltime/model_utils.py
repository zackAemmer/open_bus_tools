import os

import numpy as np
import torch

from openbustools.traveltime.models import embedding, transformer
from openbustools.traveltime.models.deeptte import DeepTTE
from openbustools.traveltime import data_loader
from openbustools import data_utils
from openbustools.traveltime.models import avg_speed, conv, ff, persistent, rnn, schedule


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


def make_model(model_type):
    # # Declare base models
    # base_model_list = []
    # base_model_list.append(avg_speed.AvgHourlySpeedModel("AVG"))
    # base_model_list.append(persistent.PersistentTimeSeqModel("PER_TIM"))
    # if not skip_gtfs:
    #     base_model_list.append(schedule.TimeTableModel("SCH"))
    # # Declare neural network models
    if model_type=="FF":
        model = ff.FF_L(
            "FF",
            n_features=8,
            collate_fn=data_loader.basic_collate_nosch,
        )
    elif model_type=="FF_GTFS":
        model = ff.FF_L(
            "FF_GTFS",
            n_features=14,
            collate_fn=data_loader.basic_collate,
        )
    elif model_type=="FF_GRID":
        model = ff.FF_GRID_L(
            "FF_GRID",
            n_features=14,
            n_grid_features=3*3*1,
            # n_grid_features=3*3*3*3,
            grid_compression_size=8,
            collate_fn=data_loader.basic_grid_collate,
        )
    elif model_type=="CONV":
        model = conv.CONV_L(
            "CONV",
            n_features=4,
            collate_fn=data_loader.sequential_collate_nosch,
        )
    elif model_type=="CONV_GTFS":
        model = conv.CONV_L(
            "CONV_GTFS",
            n_features=10,
            collate_fn=data_loader.sequential_collate,
        )
    elif model_type=="CONV_GRID":
        model = conv.CONV_GRID_L(
            "CONV_GRID",
            n_features=10,
            n_grid_features=3*3*1,
            # n_grid_features=3*3*3*3,
            grid_compression_size=8,
            collate_fn=data_loader.sequential_grid_collate,
        )
    elif model_type=="GRU":
        model = rnn.GRU_L(
            "GRU",
            input_size=4,
            collate_fn=data_loader.collate_seq,
            batch_size=embedding.HYPERPARAM_DICT['GRU']['batch_size'],
            hidden_size=embedding.HYPERPARAM_DICT['GRU']['hidden_size'],
            num_layers=embedding.HYPERPARAM_DICT['GRU']['num_layers'],
            dropout_rate=embedding.HYPERPARAM_DICT['GRU']['dropout_rate'],
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
    # # Load weights if applicable
    # if load_weights:
    #     new_base_model_list = []
    #     for b in base_model_list:
    #         new_base_model_list.append(data_utils.load_pkl(f"{weight_folder}../../../../../{b.model_name}_{fold_num}.pkl"))
    #         base_model_list = new_base_model_list
    #     last_ckpt = os.listdir(weight_folder)
    #     if not torch.cuda.is_available():
    #         model = model.load_from_checkpoint(f"{weight_folder}{last_ckpt[0]}", map_location=torch.device('cpu')).eval()
    #     else:
    #         model = model.load_from_checkpoint(f"{weight_folder}{last_ckpt[0]}").eval()
    return model


def pad_sequence(sequences, lengths):
    padded = torch.zeros(len(sequences), lengths[0]).float()
    for i, seq in enumerate(sequences):
        seq = torch.Tensor(seq)
        padded[i, :lengths[i]] = seq[:]
    return padded


def to_var(var):
    if torch.is_tensor(var):
        var = Variable(var)
        if torch.cuda.is_available():
            var = var.cuda()
        return var
    if isinstance(var, int) or isinstance(var, float):
        return var
    if isinstance(var, dict):
        for key in var:
            var[key] = to_var(var[key])
        return var
    if isinstance(var, list):
        var = list(map(lambda x: to_var(x), var))
        return var


def get_local_seq(full_seq, kernel_size, mean, std):
    seq_len = full_seq.size()[1]

    if torch.cuda.is_available():
        indices = torch.cuda.LongTensor(seq_len)
    else:
        indices = torch.LongTensor(seq_len)

    torch.arange(0, seq_len, out = indices)
    indices = Variable(indices, requires_grad = False)   ### size [max len of trip in batch, ]

    first_seq = torch.index_select(full_seq, dim = 1, index = indices[kernel_size - 1:])
    second_seq = torch.index_select(full_seq, dim = 1, index = indices[:-kernel_size + 1])

    local_seq = first_seq - second_seq   ### this is basically a lag operation feature of local path

    local_seq = (local_seq - mean) / std

    return local_seq


def extract_results(model_results, city):
    # Extract metric results
    fold_results = [x['All_Losses'] for x in model_results]
    cities = []
    models = []
    mapes = []
    rmses = []
    maes = []
    fold_nums = []
    for fold_num in range(0,len(fold_results)):
        for value in range(0,len(fold_results[0])):
            cities.append(city)
            fold_nums.append(fold_num)
            models.append(fold_results[fold_num][value][0])
            mapes.append(fold_results[fold_num][value][1])
            rmses.append(fold_results[fold_num][value][2])
            maes.append(fold_results[fold_num][value][3])
    result_df = pd.DataFrame({
        "Model": models,
        "City": cities,
        "Fold": fold_nums,
        "MAPE": mapes,
        "RMSE": rmses,
        "MAE": maes
    })
    # # Extract NN loss curves
    # loss_df = []
    # # Iterate folds
    # for fold_results in model_results:
    #     # Iterate models
    #     for model in fold_results['Loss_Curves']:
    #         for mname, loss_curves in model.items():
    #             # Iterate loss curves
    #             for lname, loss in loss_curves.items():
    #                 df = pd.DataFrame({
    #                     "City": city,
    #                     "Fold": fold_results['Fold'],
    #                     "Model": mname,
    #                     "Loss Set": lname,
    #                     "Epoch": np.arange(len(loss)),
    #                     "Loss": loss
    #                 })
    #                 loss_df.append(df)
    # loss_df = pd.concat(loss_df)
    # Extract train times
    names_df = np.array([x['Model_Names'] for x in model_results]).flatten()
    train_time_df = np.array([x['Train_Times'] for x in model_results]).flatten()
    folds_df = np.array([np.repeat(i,len(model_results[i]['Model_Names'])) for i in range(len(model_results))]).flatten()
    city_df = np.array(np.repeat(city,len(folds_df))).flatten()
    train_time_df = pd.DataFrame({
        "City": city_df,
        "Fold": folds_df,
        "Model":  names_df,
        "Time": train_time_df
    })
    return result_df, train_time_df


def extract_gen_results(gen_results, city):
    # Extract generalization results
    res = []
    experiments = ["Train_Losses","Test_Losses","Holdout_Losses","Tune_Train_Losses","Tune_Test_Losses"]
    for ex in experiments:
        fold_results = [x[ex] for x in gen_results]
        cities = []
        models = []
        mapes = []
        rmses = []
        maes = []
        fold_nums = []
        for fold_num in range(0,len(fold_results)):
            for value in range(0,len(fold_results[0])):
                cities.append(city)
                fold_nums.append(fold_num)
                models.append(fold_results[fold_num][value][0])
                mapes.append(fold_results[fold_num][value][1])
                rmses.append(fold_results[fold_num][value][2])
                maes.append(fold_results[fold_num][value][3])
        gen_df = pd.DataFrame({
            "Model": models,
            "City": cities,
            "Loss": ex,
            "Fold": fold_nums,
            "MAPE": mapes,
            "RMSE": rmses,
            "MAE": maes
        })
        res.append(gen_df)
    return pd.concat(res, axis=0)

def extract_lightning_results(model_name, base_folder, city_name):
    all_data = []
    col_names = ["train_loss_epoch","valid_loss","test_loss"]
    # for model_name in os.listdir(base_folder):
    #     model_folder = os.path.join(base_folder, model_name)
    #     if not os.path.isdir(model_folder):
    #         continue
    for fold_folder in os.listdir(base_folder):
        fold_path = os.path.join(base_folder, fold_folder)
        if not os.path.isdir(fold_path):
            continue
        metrics_file = os.path.join(fold_path, "metrics.csv")
        if not os.path.exists(metrics_file):
            continue
        # Read metrics file into a dataframe
        df = pd.read_csv(metrics_file)
        # Rename the columns to include model and fold names
        col_names_mapping = [f"{model_name}_{c}" for c in col_names]
        for i in range(len(col_names)):
            df_sub = df[["epoch", col_names_mapping[i]]].dropna()
            col_remap = {f"{col_names_mapping[i]}": "Loss", "epoch": "Epoch"}
            df_sub.rename(columns=col_remap, inplace=True)
            df_sub["Model"] = model_name
            df_sub["Loss Set"] = col_names[i]
            df_sub["Fold"] = fold_folder.split("_")[1]
            df_sub["City"] = city_name
            df_sub["Loss Set"].replace(to_replace=col_names, value=["Train","Valid","Test"], inplace=True)
            all_data.append(df_sub)
    # Concatenate all dataframes into a single dataframe
    result_df = pd.concat(all_data, axis=0)
    return result_df

def create_tensor_mask(seq_lens, device, drop_first=True):
    """
    Create a mask based on a tensor of sequence lengths.
    """
    max_len = max(seq_lens)
    mask = torch.zeros(len(seq_lens), max_len, dtype=torch.bool, device=device)
    for i, seq_len in enumerate(seq_lens):
        mask[i, :seq_len] = 1
    if drop_first:
        mask[:,0] = 0
    return mask

def pad_tensors(tensor_list, pad_dim):
    """
    Pad list of tensors with unequal lengths on pad_dim and combine.
    """
    tensor_lens = [tensor.shape[pad_dim] for tensor in tensor_list]
    max_len = max(tensor_lens)
    total_dim = len(tensor_list[0].shape)
    paddings = []
    for tensor in tensor_list:
        padding = list(0 for i in range(total_dim))
        padding[pad_dim] = max_len - tensor.shape[pad_dim]
        paddings.append(tuple(padding))
    padded_tensor_list = [torch.nn.functional.pad(tensor, paddings[i]) for i, tensor in enumerate(tensor_list)]
    padded_tensor_list = torch.cat(padded_tensor_list, dim=0)
    return padded_tensor_list

def aggregate_tts(tts, mask):
    """
    Convert a sequence of predicted travel times to total travel time.
    """
    masked_tts = (tts*mask)
    total_tts = np.sum(masked_tts, axis=1)
    return total_tts