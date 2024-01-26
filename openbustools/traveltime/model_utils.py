import os
import pickle

import numpy as np
import pandas as pd
from sklearn import metrics
import torch

from openbustools.traveltime.models import transformer
from openbustools.traveltime.models.deeptte import DeepTTE
from openbustools.traveltime import data_loader
from openbustools.traveltime.models import conv, ff, rnn


HYPERPARAM_DICT = {
    'FF': {
        'batch_size': 512,
        'hidden_size': 128,
        'num_layers': 2,
        'dropout_rate': .2,
        'grid_input_size': 3*4,
        'grid_compression_size': 16
    },
    'CONV': {
        'batch_size': 512,
        'hidden_size': 64,
        'num_layers': 3,
        'dropout_rate': .1,
        'grid_input_size': 3*4,
        'grid_compression_size': 16
    },
    'GRU': {
        'batch_size': 512,
        'hidden_size': 64,
        'num_layers': 2,
        'dropout_rate': .05,
        'grid_input_size': 3*4,
        'grid_compression_size': 16
    },
    'TRSF': {
        'batch_size': 512,
        'hidden_size': 128,
        'num_layers': 4,
        'dropout_rate': .1,
        'grid_input_size': 3*4,
        'grid_compression_size': 16
    },
    'DEEPTTE': {
        'batch_size': 512
    }
}
MODEL_ORDER = [
    'AVG',
    'PERT',
    'SCH',
    'FF','FF_TUNED',
    'FF_STATIC','FF_STATIC_TUNED',
    'FF_REALTIME','FF_REALTIME_TUNED',
    'CONV','CONV_TUNED',
    'CONV_STATIC','CONV_STATIC_TUNED',
    'CONV_REALTIME','CONV_REALTIME_TUNED',
    'GRU','GRU_TUNED',
    'GRU_STATIC','GRU_STATIC_TUNED',
    'GRU_REALTIME','GRU_REALTIME_TUNED',
    'TRSF','TRSF_TUNED',
    'TRSF_STATIC','TRSF_STATIC_TUNED',
    'TRSF_REALTIME','TRSF_REALTIME_TUNED',
    'DEEPTTE','DEEPTTE_TUNED',
    'DEEPTTE_STATIC','DEEPTTE_STATIC_TUNED',
]
EXPERIMENT_ORDER = [
    'same_city',
    'holdout',
    'diff_city'
]


def aggregate_tts(tts, mask):
    """Convert a sequence of predicted travel times to total travel time."""
    masked_tts = (tts*mask)
    total_tts = torch.sum(masked_tts, dim=0)
    return total_tts


def fill_tensor_mask(mask, x_sl, drop_first=True):
    """Fill a mask based on sequence lengths."""
    for i, seq_len in enumerate(x_sl):
        mask[:seq_len, i] = 1
    if drop_first:
        mask[0,:] = 0
    return mask


def basic_train_step(model, batch):
    _x, _y, seq_lens = batch
    labels, labels_agg = _y
    out_agg = model.forward(_x, seq_lens)
    loss = model.loss_fn(out_agg, labels_agg)
    return loss


def basic_pred_step(model, batch):
    _x, _y, seq_lens = batch
    labels, labels_agg = _y
    out_agg = model.forward(_x, seq_lens)
    out_agg = data_loader.denormalize(out_agg, model.config['cumul_time_s'])
    labels_agg = data_loader.denormalize(labels_agg, model.config['cumul_time_s'])
    return {'preds': out_agg.detach().cpu().numpy(), 'labels': labels_agg.detach().cpu().numpy()}


def seq_train_step(model, batch):
    _x, _y, seq_lens = batch
    labels, labels_agg = _y
    out = model.forward(_x)
    # Masked loss; init mask here since module knows device
    mask = torch.zeros(max(seq_lens), len(seq_lens), dtype=torch.bool, device=model.device)
    mask = fill_tensor_mask(mask, seq_lens)
    out = out[mask]
    labels = labels[mask]
    loss = model.loss_fn(out, labels)
    return loss


def seq_pred_step(model, batch):
    _x, _y, seq_lens = batch
    labels, labels_agg = _y
    out = model.forward(_x)
    mask = torch.zeros(max(seq_lens), len(seq_lens), dtype=torch.bool, device=model.device)
    mask = fill_tensor_mask(mask, seq_lens)
    out = data_loader.denormalize(out, model.config['calc_time_s'])
    out_agg = aggregate_tts(out, mask)
    labels = data_loader.denormalize(labels, model.config['calc_time_s'])
    labels_agg = aggregate_tts(labels, mask)
    return {'preds': out_agg.detach().cpu().numpy(), 'labels': labels_agg.detach().cpu().numpy(), 'preds_raw': out.detach().cpu().numpy(), 'labels_raw': labels.detach().cpu().numpy(), 'mask': mask.detach().cpu().numpy()}


def load_results(res_folder):
    all_res = []
    all_out = []
    for model_res_file in os.listdir(res_folder):
        if model_res_file.split('.')[-1]=='pkl':
            res, out = format_model_res(f"{res_folder}{model_res_file}")
            all_res.append(res)
            all_out.append(out)
    all_res = pd.concat(all_res)
    all_out = pd.concat(all_out)
    all_res['model_archetype'] = all_res['model'].str.split('_').str[0]
    all_res['is_tuned'] = False
    all_res.loc[all_res['model'].str.split('_').str[-1]=='TUNED', 'is_tuned'] = True
    all_res['plot_order_model'] = all_res['model'].apply(lambda x: MODEL_ORDER.index(x))
    all_res['plot_order_experiment'] = all_res['experiment_name'].apply(lambda x: EXPERIMENT_ORDER.index(x))
    all_res = all_res.sort_values(['plot_order_model','plot_order_experiment'])
    return (all_res, all_out)


def format_model_res(model_res_file):
    model_res = pickle.load(open(model_res_file, 'rb'))
    all_res = []
    all_outputs = []
    for fold_num, experiment_res in model_res.items():
        for experiment_name, preds_and_labels in experiment_res.items():
            reg = performance_metrics(preds_and_labels['labels'], preds_and_labels['preds'])
            res_df = pd.DataFrame({
                'model': model_res_file.split('.')[-2].split('/')[-1],
                'experiment_name': experiment_name,
                'fold': fold_num,
                'metric': list(reg.keys()),
                'value': list(reg.values())
            })
            out_df = pd.DataFrame({
                'model': model_res_file.split('.')[-2].split('/')[-1],
                'experiment_name': experiment_name,
                'fold': fold_num,
                'preds': preds_and_labels['preds'],
                'labels': preds_and_labels['labels'],
                'residuals': preds_and_labels['labels'] - preds_and_labels['preds']
            })
            all_res.append(res_df)
            all_outputs.append(out_df)
    all_res = pd.concat(all_res).reset_index(drop=True)
    all_outputs = pd.concat(all_outputs).reset_index(drop=True)
    return (all_res, all_outputs)


def performance_metrics(labels, preds, print_res=False):
    # Summary statistics
    label_min = min(labels)
    label_max = max(labels)
    pred_min = min(preds)
    pred_max = max(preds)
    label_mean = np.mean(labels)
    pred_mean = np.mean(preds)
    # Performance metrics
    mae = metrics.mean_absolute_error(labels, preds)
    mse = metrics.mean_squared_error(labels, preds)
    mape = metrics.mean_absolute_percentage_error(labels, preds)
    ev = metrics.explained_variance_score(labels, preds)
    r2 = metrics.r2_score(labels, preds)
    if print_res:
        print(f"Label min: {round(label_min,1)}, mean: {round(label_mean,1)}, max: {round(label_max,1)}")
        print(f"Pred min: {round(pred_min,1)}, mean: {round(pred_mean,1)}, max: {round(pred_max,1)}")
        print(f"MAE: {round(mae)}")
        print(f"RMSE: {round(np.sqrt(mse))}")
        print(f"MAPE: {round(mape,3)}")
        print(f"Explained Variance: {round(ev,3)}")
        print(f"R2 Score: {round(r2,3)}")
    return {
        'label_min': label_min,
        'label_max': label_max,
        'label_mean': label_mean,
        'pred_min': pred_min,
        'pred_max': pred_max,
        'pred_mean': pred_mean,
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
            input_size=5,
            collate_fn=data_loader.collate_seq,
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
            input_size=9,
            collate_fn=data_loader.collate_seq_static,
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
            input_size=9,
            collate_fn=data_loader.collate_seq_realtime,
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
            input_size=5,
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
            input_size=9,
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
            input_size=9,
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
            input_size=5,
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
            input_size=9,
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
            input_size=9,
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
            input_size=5,
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
            input_size=9,
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
            input_size=9,
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
            input_size=5,
            collate_fn=data_loader.collate_deeptte,
            batch_size=HYPERPARAM_DICT[model_archetype]['batch_size'],
        )
    elif model_type=="DEEPTTE_STATIC":
        model = DeepTTE.Net(
            f"DEEPTTE_STATIC-{fold_num}",
            config=config,
            holdout_routes=holdout_routes,
            input_size=9,
            collate_fn=data_loader.collate_deeptte_static,
            batch_size=HYPERPARAM_DICT[model_archetype]['batch_size'],
        )
    return model


def load_model(model_folder, network_name, model_type, fold_num):
    """Load latest checkpoint depending on user chosen model type and fold."""
    last_version = str(sorted([int(x.split('_')[1]) for x in os.listdir(f"{model_folder}{network_name}/{model_type}-{fold_num}")])[-1])
    last_version = f"version_{last_version}"
    last_ckpt = sorted(os.listdir(f"{model_folder}{network_name}/{model_type}-{fold_num}/{last_version}/checkpoints/"))[-1]
    model_archetype = model_type.split('_')
    model_archetype = list(filter(lambda a: a != 'TUNED', model_archetype))
    model_archetype = '_'.join(model_archetype[:2])
    if model_archetype in ['FF', 'FF_STATIC']:
        model_cl = ff.FF
    elif model_archetype=='FF_REALTIME':
        model_cl = ff.FFRealtime
    elif model_archetype in ['GRU', 'GRU_STATIC']:
        model_cl = rnn.GRU
    elif model_archetype=='GRU_REALTIME':
        model_cl = rnn.GRURealtime
    elif model_archetype in ['CONV', 'CONV_STATIC']:
        model_cl = conv.CONV
    elif model_archetype=='CONV_REALTIME':
        model_cl = conv.CONVRealtime
    elif model_archetype in ['TRSF', 'TRSF_STATIC']:
        model_cl = transformer.TRSF
    elif model_archetype=='TRSF_REALTIME':
        model_cl = transformer.TRSFRealtime
    elif model_archetype in ['DEEPTTE', 'DEEPTTE_STATIC']:
        model_cl = DeepTTE.Net
    try:
        model = model_cl.load_from_checkpoint(f"{model_folder}{network_name}/{model_type}-{fold_num}/{last_version}/checkpoints/{last_ckpt}", strict=False).eval()
    except RuntimeError:
        model = model_cl.load_from_checkpoint(f"{model_folder}{network_name}/{model_type}-{fold_num}/{last_version}/checkpoints/{last_ckpt}", strict=False, map_location=torch.device('cpu')).eval()
    return model