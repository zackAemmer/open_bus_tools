import os

import numpy as np
from sklearn import metrics
import torch

from openbustools.traveltime.models import transformer
from openbustools.traveltime.models.deeptte import DeepTTE
from openbustools.traveltime import data_loader
from openbustools.traveltime.models import avg_speed, conv, ff, persistent, rnn, schedule

HYPERPARAM_DICT = {
    'FF': {
        'batch_size': 64,
        'hidden_size': 32,
        'num_layers': 2,
        'dropout_rate': .1,
        'grid_input_size': 8*4,
        'grid_compression_size': 16
    },
    'CONV': {
        'batch_size': 64,
        'hidden_size': 32,
        'num_layers': 2,
        'dropout_rate': .1,
        'grid_input_size': 8*4,
        'grid_compression_size': 16
    },
    'GRU': {
        'batch_size': 64,
        'hidden_size': 32,
        'num_layers': 2,
        'dropout_rate': .1,
        'grid_input_size': 8*4,
        'grid_compression_size': 16
    },
    'TRSF': {
        'batch_size': 64,
        'hidden_size': 32,
        'num_layers': 2,
        'dropout_rate': .1,
        'grid_input_size': 8*4,
        'grid_compression_size': 16
    },
    'DEEPTTE': {
        'batch_size': 64
    }
}


def aggregate_tts(tts, mask):
    """Convert a sequence of predicted travel times to total travel time."""
    masked_tts = (tts*mask)
    total_tts = np.sum(masked_tts, axis=1)
    return total_tts


def fill_tensor_mask(mask, x_sl, drop_first=True):
    """Fill a mask based on sequence lengths."""
    for i, seq_len in enumerate(x_sl):
        mask[:seq_len, i] = 1
    if drop_first:
        mask[0,:] = 0
    return mask


def performance_metrics(labels, preds, print_res=False):
    # Summary statistics
    label_min=min(labels)
    label_max=max(labels)
    pred_min=min(preds)
    pred_max=max(preds)
    label_mean=np.mean(labels)
    pred_mean=np.mean(preds)
    label_med=np.median(labels)
    pred_med=np.median(preds)
    # Performance metrics
    mae=metrics.mean_absolute_error(labels, preds)
    mse=metrics.mean_squared_error(labels, preds)
    mape=metrics.mean_absolute_percentage_error(labels, preds)
    ev=metrics.explained_variance_score(labels, preds)
    r2=metrics.r2_score(labels, preds)
    if print_res:
        print(f"Label min: {round(label_min,1)}, mean: {round(label_mean,1)}, median: {round(label_med,1)}, max: {round(label_max,1)}")
        print(f"Pred min: {round(pred_min,1)}, mean: {round(pred_mean,1)}, median: {round(pred_med,1)}, max: {round(pred_max,1)}")
        print(f"MAE: {round(mae)}")
        print(f"RMSE: {round(np.sqrt(mse))}")
        print(f"MAPE: {round(mape,3)}")
        print(f"Explained Variance: {round(ev,3)}")
        print(f"R2 Score: {round(r2,3)}")
    return {
        'label_min': label_min,
        'label_max': label_max,
        'label_mean': label_mean,
        'label_med': label_med,
        'pred_min': pred_min,
        'pred_max': pred_max,
        'pred_mean': pred_mean,
        'pred_med': pred_med,
        'mae': mae,
        'rmse': np.sqrt(mae),
        'mape': mape,
        'ex_var': ev,
        'r_score': r2
    }


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


def make_model(model_type, fold_num, config, holdout_routes=None):
    """Allow one main script to be re-used for different model types."""
    model_archetype = model_type.split('_')[0]
    if model_type=="FF":
        model = ff.FF(
            f"FF-{fold_num}",
            config=config,
            holdout_routes=holdout_routes,
            input_size=8,
            collate_fn=data_loader.collate,
            batch_size=HYPERPARAM_DICT[model_archetype]['batch_size'],
            hidden_size=HYPERPARAM_DICT[model_archetype]['hidden_size'],
            num_layers=HYPERPARAM_DICT[model_archetype]['num_layers'],
            dropout_rate=HYPERPARAM_DICT[model_archetype]['dropout_rate'],
        )
    elif model_type=="FF_STATIC":
        model = ff.FF(
            f"FF_STATIC-{fold_num}",
            config=config,
            holdout_routes=holdout_routes,
            input_size=16,
            collate_fn=data_loader.collate_static,
            batch_size=HYPERPARAM_DICT[model_archetype]['batch_size'],
            hidden_size=HYPERPARAM_DICT[model_archetype]['hidden_size'],
            num_layers=HYPERPARAM_DICT[model_archetype]['num_layers'],
            dropout_rate=HYPERPARAM_DICT[model_archetype]['dropout_rate'],
        )
    elif model_type=="FF_REALTIME":
        model = ff.FFRealtime(
            f"FF_REALTIME-{fold_num}",
            config=config,
            holdout_routes=holdout_routes,
            input_size=16,
            collate_fn=data_loader.collate_realtime,
            batch_size=HYPERPARAM_DICT[model_archetype]['batch_size'],
            hidden_size=HYPERPARAM_DICT[model_archetype]['hidden_size'],
            num_layers=HYPERPARAM_DICT[model_archetype]['num_layers'],
            dropout_rate=HYPERPARAM_DICT[model_archetype]['dropout_rate'],
            grid_input_size=HYPERPARAM_DICT[model_archetype]['grid_input_size'],
            grid_compression_size=HYPERPARAM_DICT[model_archetype]['grid_compression_size']
        )
    elif model_type=="CONV":
        model = conv.CONV(
            f"CONV-{fold_num}",
            config=config,
            holdout_routes=holdout_routes,
            input_size=4,
            collate_fn=data_loader.collate_seq,
            batch_size=HYPERPARAM_DICT[model_archetype]['batch_size'],
            hidden_size=HYPERPARAM_DICT[model_archetype]['hidden_size'],
            num_layers=HYPERPARAM_DICT[model_archetype]['num_layers'],
            dropout_rate=HYPERPARAM_DICT[model_archetype]['dropout_rate'],
        )
    elif model_type=="CONV_STATIC":
        model = conv.CONV(
            f"CONV_STATIC-{fold_num}",
            config=config,
            holdout_routes=holdout_routes,
            input_size=8,
            collate_fn=data_loader.collate_seq_static,
            batch_size=HYPERPARAM_DICT[model_archetype]['batch_size'],
            hidden_size=HYPERPARAM_DICT[model_archetype]['hidden_size'],
            num_layers=HYPERPARAM_DICT[model_archetype]['num_layers'],
            dropout_rate=HYPERPARAM_DICT[model_archetype]['dropout_rate'],
        )
    elif model_type=="CONV_REALTIME":
        model = conv.CONVRealtime(
            f"CONV_REALTIME-{fold_num}",
            config=config,
            holdout_routes=holdout_routes,
            input_size=8,
            collate_fn=data_loader.collate_seq_realtime,
            batch_size=HYPERPARAM_DICT[model_archetype]['batch_size'],
            hidden_size=HYPERPARAM_DICT[model_archetype]['hidden_size'],
            num_layers=HYPERPARAM_DICT[model_archetype]['num_layers'],
            dropout_rate=HYPERPARAM_DICT[model_archetype]['dropout_rate'],
            grid_input_size=HYPERPARAM_DICT[model_archetype]['grid_input_size'],
            grid_compression_size=HYPERPARAM_DICT[model_archetype]['grid_compression_size']
        )
    elif model_type=="GRU":
        model = rnn.GRU(
            f"GRU-{fold_num}",
            config=config,
            holdout_routes=holdout_routes,
            input_size=4,
            collate_fn=data_loader.collate_seq,
            batch_size=HYPERPARAM_DICT[model_archetype]['batch_size'],
            hidden_size=HYPERPARAM_DICT[model_archetype]['hidden_size'],
            num_layers=HYPERPARAM_DICT[model_archetype]['num_layers'],
            dropout_rate=HYPERPARAM_DICT[model_archetype]['dropout_rate'],
        )
    elif model_type=="GRU_STATIC":
        model = rnn.GRU(
            f"GRU_STATIC-{fold_num}",
            config=config,
            holdout_routes=holdout_routes,
            input_size=8,
            collate_fn=data_loader.collate_seq_static,
            batch_size=HYPERPARAM_DICT[model_archetype]['batch_size'],
            hidden_size=HYPERPARAM_DICT[model_archetype]['hidden_size'],
            num_layers=HYPERPARAM_DICT[model_archetype]['num_layers'],
            dropout_rate=HYPERPARAM_DICT[model_archetype]['dropout_rate'],
        )
    elif model_type=="GRU_REALTIME":
        model = rnn.GRURealtime(
            f"GRU_REALTIME-{fold_num}",
            config=config,
            holdout_routes=holdout_routes,
            input_size=8,
            collate_fn=data_loader.collate_seq_realtime,
            batch_size=HYPERPARAM_DICT[model_archetype]['batch_size'],
            hidden_size=HYPERPARAM_DICT[model_archetype]['hidden_size'],
            num_layers=HYPERPARAM_DICT[model_archetype]['num_layers'],
            dropout_rate=HYPERPARAM_DICT[model_archetype]['dropout_rate'],
            grid_input_size=HYPERPARAM_DICT[model_archetype]['grid_input_size'],
            grid_compression_size=HYPERPARAM_DICT[model_archetype]['grid_compression_size']
        )
    elif model_type=="TRSF":
        model = transformer.TRSF(
            f"TRSF-{fold_num}",
            config=config,
            holdout_routes=holdout_routes,
            input_size=4,
            collate_fn=data_loader.collate_seq,
            batch_size=HYPERPARAM_DICT[model_archetype]['batch_size'],
            hidden_size=HYPERPARAM_DICT[model_archetype]['hidden_size'],
            num_layers=HYPERPARAM_DICT[model_archetype]['num_layers'],
            dropout_rate=HYPERPARAM_DICT[model_archetype]['dropout_rate'],
        )
    elif model_type=="TRSF_STATIC":
        model = transformer.TRSF(
            f"TRSF_STATIC-{fold_num}",
            config=config,
            holdout_routes=holdout_routes,
            input_size=8,
            collate_fn=data_loader.collate_seq_static,
            batch_size=HYPERPARAM_DICT[model_archetype]['batch_size'],
            hidden_size=HYPERPARAM_DICT[model_archetype]['hidden_size'],
            num_layers=HYPERPARAM_DICT[model_archetype]['num_layers'],
            dropout_rate=HYPERPARAM_DICT[model_archetype]['dropout_rate'],
        )
    elif model_type=="TRSF_REALTIME":
        model = transformer.TRSFRealtime(
            f"TRSF_REALTIME-{fold_num}",
            config=config,
            holdout_routes=holdout_routes,
            input_size=8,
            collate_fn=data_loader.collate_seq_realtime,
            batch_size=HYPERPARAM_DICT[model_archetype]['batch_size'],
            hidden_size=HYPERPARAM_DICT[model_archetype]['hidden_size'],
            num_layers=HYPERPARAM_DICT[model_archetype]['num_layers'],
            dropout_rate=HYPERPARAM_DICT[model_archetype]['dropout_rate'],
            grid_input_size=HYPERPARAM_DICT[model_archetype]['grid_input_size'],
            grid_compression_size=HYPERPARAM_DICT[model_archetype]['grid_compression_size']
        )
    elif model_type=="DEEPTTE":
        model = DeepTTE.Net(
            f"DEEPTTE-{fold_num}",
            config=config,
            holdout_routes=holdout_routes,
            input_size=4,
            collate_fn=data_loader.collate_deeptte,
            batch_size=HYPERPARAM_DICT[model_archetype]['batch_size'],
        )
    elif model_type=="DEEPTTE_STATIC":
        model = DeepTTE.Net(
            f"DEEPTTE_STATIC-{fold_num}",
            config=config,
            holdout_routes=holdout_routes,
            input_size=8,
            collate_fn=data_loader.collate_deeptte_static,
            batch_size=HYPERPARAM_DICT[model_archetype]['batch_size'],
        )
    return model


def load_model(model_folder, network_name, model_type, fold_num):
    """Load latest checkpoint depending on user chosen model type and fold."""
    last_version = str(sorted([int(x.split('_')[1]) for x in os.listdir(f"{model_folder}{network_name}/{model_type}_{fold_num}")])[-1])
    last_version = f"version_{last_version}"
    last_ckpt = sorted(os.listdir(f"{model_folder}{network_name}/{model_type}_{fold_num}/{last_version}/checkpoints/"))[-1]
    if model_type=='FF':
        model_cl = ff.FF
    elif model_type=='FF_REALTIME':
        model_cl = ff.FFRealtime
    elif model_type=='GRU':
        model_cl = rnn.GRU
    elif model_type=='GRU_REALTIME':
        model_cl = rnn.GRURealtime
    elif model_type=='CONV':
        model_cl = conv.CONV
    elif model_type=='CONV_REALTIME':
        model_cl = conv.CONVRealtime
    elif model_type=='TRSF':
        model_cl = transformer.TRSF
    elif model_type=='TRSF_REALTIME':
        model_cl = transformer.TRSFRealtime
    elif model_type=='DEEPTTE':
        model_cl = DeepTTE.Net
    try:
        model = model_cl.load_from_checkpoint(f"{model_folder}{network_name}/{model_type}_{fold_num}/{last_version}/checkpoints/{last_ckpt}", strict=False).eval()
    except:
        model = model_cl.load_from_checkpoint(f"{model_folder}{network_name}/{model_type}_{fold_num}/{last_version}/checkpoints/{last_ckpt}", strict=False, map_location=torch.device('cpu')).eval()
    return model