from itertools import compress

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from openbustools.traveltime import data_loader

LABEL_FEATS = [
    "calc_time_s",
    "cumul_time_s",
]
GPS_FEATS = [
    "calc_dist_m",
    "calc_bear_d",
    "x_cent",
    "y_cent",
]
STATIC_FEATS = [
    "calc_stop_dist_m",
    "sch_time_s",
    "pass_stops_n",
    "cumul_pass_stops_n",
]
DEEPTTE_FEATS = [
    "cumul_dist_m",
]
EMBED_FEATS = [
    "t_min_of_day",
    "t_day_of_week",
]
MISC_CON_FEATS = [
    "x",
    "y",
    "locationtime",
    "calc_speed_m_s",
    "t_hour",
    "t_min",
]
MISC_CAT_FEATS = [
    "route_id",
]
HOLDOUT_ROUTES = [
    'ATB:Line:2_87','ATB:Line:2_72','ATB:Line:2_9','ATB:Line:2_5111',
    '102736','102628','102555','100129','102719','100229'
]

NUM_FEAT_COLS = LABEL_FEATS+GPS_FEATS+STATIC_FEATS+DEEPTTE_FEATS+EMBED_FEATS+MISC_CON_FEATS


def normalize(x, config_entry):
    mean = config_entry[0]
    std = config_entry[1]
    return (x - mean) / std


def denormalize(x, config_entry):
    mean = config_entry[0]
    std = config_entry[1]
    return x * std + mean


def create_config(data, idxs):
    ext_data = [data[x]['feats_n'] for x in idxs]
    config = {}
    for i,col in enumerate(NUM_FEAT_COLS):
        col_data = np.concatenate([samp[:,i] for samp in ext_data])
        col_mean = np.mean(col_data)
        col_sd = np.std(col_data)
        config[col] = (col_mean, col_sd)
    return config


def normalize_samples(data, config):
    for data_idx in list(data.keys()):
        normed = [normalize(data[data_idx]['feats_n'][:,i], config[feat_name]) for i,feat_name in enumerate(NUM_FEAT_COLS)]
        data[data_idx]['feats_n_norm'] = np.stack(normed, axis=1)


def load_h5(data_folders, dates, only_holdout=False, **kwargs):
    """Preload all samples from an h5py file into memory, outside of dataset."""
    data = {}
    # Set holdout routes, variable names
    if 'holdout_routes' in kwargs:
        holdout_routes = kwargs['holdout_routes']
    else:
        holdout_routes = []
    current_max_key = 0
    # Load all of the folder/day data into one dictionary, reindexing along way
    for data_folder in data_folders:
        with h5py.File(f"{data_folder}/samples.hdf5", 'r') as f:
            for day in dates:
                try:
                    sids, sidxs = np.unique(f[day]['shingle_ids'], return_index=True)
                    feats_n = np.split(f[day]['feats_n'], sidxs[1:], axis=0)
                    feats_g = np.split(f[day]['feats_g'], sidxs[1:], axis=0)
                    feats_c = np.split(f[day]['feats_c'], sidxs[1:], axis=0)
                    # Keep either all but, or only samples from holdout routes
                    if len(holdout_routes)>0:
                        is_not_holdout = np.array([x[0].astype(str)[0] not in holdout_routes for x in feats_c])
                        if only_holdout:
                            is_not_holdout = np.invert(is_not_holdout)
                        sids = sids[is_not_holdout]
                        feats_n = list(compress(feats_n, is_not_holdout))
                        feats_g = list(compress(feats_g, is_not_holdout))
                        feats_c = list(compress(feats_c, is_not_holdout))
                    sids = np.arange(current_max_key, current_max_key+len(sids))
                    sample = {fs: {'feats_n': fn,'feats_g': fg,'feats_c': fc} for fs,fn,fg,fc in zip(sids,feats_n,feats_g,feats_c)}
                    data.update(sample)
                    current_max_key = sorted(data.keys())[-1]+1
                except KeyError:
                    print(f"Day not found: {day}")
    # Add sample key to all data entries that has normalized data
    if 'config' in kwargs:
        config = kwargs['config']
    else:
        config = create_config(data, list(data.keys()))
    normalize_samples(data, config)
    return (data, holdout_routes, config)


class H5Dataset(Dataset):
    """Provide samples by indexing dict."""
    def __init__(self, data):
        self.data = data
        self.include_grid = False
    def __getitem__(self, index):
        if not self.include_grid:
            return (self.data[index]['feats_n_norm'], self.data[index]['feats_n'])
        else:
            return (self.data[index]['feats_n_norm'], self.data[index]['feats_n'], self.data[index]['feats_g'])
    def __len__(self):
        return len(self.data)
    def to_df(self):
        shingle_ids = np.arange(len(self.data))
        shingle_lens = np.array([self.data[i]['feats_n'].shape[0] for i in np.arange(len(self))])
        shingle_ids = shingle_ids.repeat(shingle_lens)
        data = np.concatenate([self.data[i]['feats_n'] for i in np.arange(len(self))])
        data_df = pd.DataFrame(data, columns=data_loader.NUM_FEAT_COLS)
        data_df['shingle_id'] = shingle_ids
        return data_df


def collate_seq(batch):
    norm = torch.nn.utils.rnn.pad_sequence([torch.tensor(b[0], dtype=torch.float) for b in batch])
    no_norm = torch.nn.utils.rnn.pad_sequence([torch.tensor(b[1], dtype=torch.long) for b in batch])
    X_em = no_norm[:,:,[NUM_FEAT_COLS.index(x) for x in EMBED_FEATS]]
    X_em = X_em[0,:,:].unsqueeze(0).repeat(X_em.shape[0],1,1)
    X_ct = norm[:,:,[NUM_FEAT_COLS.index(x) for x in GPS_FEATS]]
    X_sl = torch.tensor([len(b[0]) for b in batch], dtype=torch.long)
    Y = norm[:,:,NUM_FEAT_COLS.index('calc_time_s')]
    Y_agg = norm[:,:,NUM_FEAT_COLS.index('cumul_time_s')]
    Y_no_norm = no_norm[:,:,NUM_FEAT_COLS.index('calc_time_s')]
    Y_agg_no_norm = no_norm[:,:,NUM_FEAT_COLS.index('cumul_time_s')]
    return ((X_em, X_ct), (Y, Y_no_norm, Y_agg, Y_agg_no_norm), X_sl)


def collate_seq_static(batch):
    norm = torch.nn.utils.rnn.pad_sequence([torch.tensor(b[0], dtype=torch.float) for b in batch])
    no_norm = torch.nn.utils.rnn.pad_sequence([torch.tensor(b[1], dtype=torch.long) for b in batch])
    X_em = no_norm[:,:,[NUM_FEAT_COLS.index(x) for x in EMBED_FEATS]]
    X_em = X_em[0,:,:].unsqueeze(0).repeat(X_em.shape[0],1,1)
    X_ct = norm[:,:,[NUM_FEAT_COLS.index(x) for x in GPS_FEATS+STATIC_FEATS]]
    X_sl = torch.tensor([len(b[0]) for b in batch], dtype=torch.long)
    Y = norm[:,:,NUM_FEAT_COLS.index('calc_time_s')]
    Y_agg = norm[:,:,NUM_FEAT_COLS.index('cumul_time_s')]
    Y_no_norm = no_norm[:,:,NUM_FEAT_COLS.index('calc_time_s')]
    Y_agg_no_norm = no_norm[:,:,NUM_FEAT_COLS.index('cumul_time_s')]
    return  ((X_em, X_ct), (Y, Y_no_norm, Y_agg, Y_agg_no_norm), X_sl)


def collate_seq_realtime(batch):
    norm = torch.nn.utils.rnn.pad_sequence([torch.tensor(b[0], dtype=torch.float) for b in batch])
    no_norm = torch.nn.utils.rnn.pad_sequence([torch.tensor(b[1], dtype=torch.long) for b in batch])
    X_em = no_norm[:,:,[NUM_FEAT_COLS.index(x) for x in EMBED_FEATS]]
    X_em = X_em[0,:,:].unsqueeze(0).repeat(X_em.shape[0],1,1)
    X_ct = norm[:,:,[NUM_FEAT_COLS.index(x) for x in GPS_FEATS+STATIC_FEATS]]
    X_sl = torch.tensor([len(b[0]) for b in batch], dtype=torch.long)
    Y = norm[:,:,NUM_FEAT_COLS.index('calc_time_s')]
    Y_agg = norm[:,:,NUM_FEAT_COLS.index('cumul_time_s')]
    Y_no_norm = no_norm[:,:,NUM_FEAT_COLS.index('calc_time_s')]
    Y_agg_no_norm = no_norm[:,:,NUM_FEAT_COLS.index('cumul_time_s')]
    X_gr = torch.nn.utils.rnn.pad_sequence([torch.tensor(b[2], dtype=torch.float) for b in batch], batch_first=True)
    X_gr = torch.swapaxes(X_gr, 1, 3)
    X_gr = torch.swapaxes(X_gr, 1, 2)
    return  ((X_em, X_ct, X_gr), (Y, Y_no_norm, Y_agg, Y_agg_no_norm), X_sl)


# def collate_deeptte(batch):
#     stat_attrs = ['cumul_dist_m', 'cumul_time_s']
#     info_attrs = ['t_day_of_week', 't_min_of_day']
#     traj_attrs = GPS_FEATS
#     attr, traj = {}, {}
#     X_sl = torch.tensor([len(b['calc_time_s']) for b in batch], dtype=torch.long)
#     traj['X_sl'] = X_sl
#     traj['calc_time_s'] = torch.nn.utils.rnn.pad_sequence([torch.tensor(b['calc_time_s'], dtype=torch.float) for b in batch])
#     labels = torch.tensor([b['cumul_time_s_no_norm'][-1] for b in batch], dtype=torch.float)
#     for k in stat_attrs:
#         attr[k] = torch.tensor([b[k][-1] for b in batch], dtype=torch.float)
#     for k in info_attrs:
#         attr[k] = torch.tensor([b[k] for b in batch], dtype=torch.long)
#     for k in traj_attrs:
#         traj[k] = torch.nn.utils.rnn.pad_sequence([torch.tensor(b[k], dtype=torch.float) for b in batch])
#     return (attr, traj, labels)


# def collate_deeptte_static(batch):
#     stat_attrs = ['cumul_dist_m', 'cumul_time_s']
#     info_attrs = ['t_day_of_week', 't_min_of_day']
#     traj_attrs = GPS_FEATS+STATIC_FEATS
#     attr, traj = {}, {}
#     X_sl = torch.tensor([len(b['calc_time_s']) for b in batch], dtype=torch.long)
#     traj['X_sl'] = X_sl
#     traj['calc_time_s'] = torch.nn.utils.rnn.pad_sequence([torch.tensor(b['calc_time_s'], dtype=torch.float) for b in batch])
#     labels = torch.tensor([b['cumul_time_s_no_norm'][-1] for b in batch], dtype=torch.float)
#     for k in stat_attrs:
#         attr[k] = torch.tensor([b[k][-1] for b in batch], dtype=torch.float)
#     for k in info_attrs:
#         attr[k] = torch.tensor([b[k] for b in batch], dtype=torch.long)
#     for k in traj_attrs:
#         traj[k] = torch.nn.utils.rnn.pad_sequence([torch.tensor(b[k], dtype=torch.float) for b in batch])
#     return (attr, traj, labels)