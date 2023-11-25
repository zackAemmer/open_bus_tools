from itertools import compress
import gc
import pickle

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

LABEL_FEATS = [
    "calc_time_s",
    "cumul_time_s",
]
EMBED_FEATS = [
    "t_min_of_day",
    "t_day_of_week",
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


def normalize(x, config_entry):
    mean = config_entry[0]
    std = config_entry[1]
    return (x - mean) / std


def denormalize(x, config_entry):
    mean = config_entry[0]
    std = config_entry[1]
    return x * std + mean


def create_config(dataset, idxs):
    data = [dataset.data[x]['feats_n'] for x in idxs]
    config = {}
    for i,col in enumerate(dataset.colnames):
        col_data = np.concatenate([samp[:,i] for samp in data])
        col_mean = np.mean(col_data)
        col_sd = np.std(col_data)
        config[col] = (col_mean, col_sd)
    return config


class H5Dataset(Dataset):
    """Load all data into memory as dataframe, provide samples by indexing groups."""
    def __init__(self, data_folders, dates, only_holdout=False, **kwargs):
        if 'holdout_routes' in kwargs:
            self.holdout_routes = kwargs['holdout_routes']
        else:
            self.holdout_routes = []
        self.data = {}
        self.colnames = LABEL_FEATS+EMBED_FEATS+GPS_FEATS+STATIC_FEATS+DEEPTTE_FEATS+MISC_CON_FEATS
        current_max_key = 0
        for data_folder in data_folders:
            with h5py.File(f"{data_folder}/samples.hdf5", 'r') as f:
                for day in dates:
                    sids, sidxs = np.unique(f[day]['shingle_ids'], return_index=True)
                    feats_n = np.split(f[day]['feats_n'], sidxs[1:], axis=0)
                    feats_g = np.split(f[day]['feats_g'], sidxs[1:], axis=0)
                    feats_c = np.split(f[day]['feats_c'], sidxs[1:], axis=0)
                    if len(self.holdout_routes)>0:
                        is_not_holdout = np.array([x[0].astype(str)[0] not in self.holdout_routes for x in feats_c])
                        if only_holdout:
                            is_not_holdout = np.invert(is_not_holdout)
                        sids = sids[is_not_holdout]
                        feats_n = list(compress(feats_n, is_not_holdout))
                        feats_g = list(compress(feats_g, is_not_holdout))
                        feats_c = list(compress(feats_c, is_not_holdout))
                    sids = np.arange(current_max_key, current_max_key+len(sids))
                    sample = {fs:{'feats_n':fn,'feats_g':fg,'feats_c':fc} for fs,fn,fg,fc in zip(sids,feats_n,feats_g,feats_c)}
                    self.data.update(sample)
                    current_max_key = sorted(self.data.keys())[-1]+1
        # Normalize the data: TODO: Allow passing config from model for models already created
        self.config = create_config(self, list(self.data.keys()))
        for data_idx in list(self.data.keys()):
            sample_data = self.data[data_idx]['feats_n']
            self.data[data_idx]['sample'] = {}
            for k in LABEL_FEATS:
                self.data[data_idx]['sample'][f"{k}_no_norm"] = sample_data[:,self.colnames.index(k)].astype(float)
            for k in LABEL_FEATS+GPS_FEATS+STATIC_FEATS+DEEPTTE_FEATS+MISC_CON_FEATS:
                self.data[data_idx]['sample'][k] = normalize(sample_data[:,self.colnames.index(k)].astype(float), self.config[k])
            for k in EMBED_FEATS:
                self.data[data_idx]['sample'][k] = sample_data[:,self.colnames.index(k)].astype(int)[0]
        self.include_grid = False
    def __getitem__(self, index):
        sample = self.data[index]['sample']
        # sample_data = s['feats_n']
        # sample_grid = s['feats_g']
        # sample = {}
        # for i,k in enumerate(LABEL_FEATS):
        #     sample[f"{k}_no_norm"] = sample_data[:,self.colnames.index(k)].astype(float)
        # for i,k in enumerate(LABEL_FEATS+GPS_FEATS+STATIC_FEATS+DEEPTTE_FEATS+MISC_CON_FEATS):
        #     sample[k] = normalize(sample_data[:,self.colnames.index(k)].astype(float), self.config[k])
        # for k in EMBED_FEATS:
        #     sample[k] = sample_data[:,self.colnames.index(k)].astype(int)[0]
        if self.include_grid:
            sample['grid'] = self.data[index]['feats_g']
        return sample
    def __len__(self):
        return len(self.data)


def collate(batch):
    y = torch.tensor([b['cumul_time_s'][-1] for b in batch], dtype=torch.float)
    y_no_norm = torch.tensor([b['cumul_time_s_no_norm'][-1] for b in batch], dtype=torch.float)
    X_em_dow = torch.tensor([b['t_day_of_week'] for b in batch], dtype=torch.int).unsqueeze(-1)
    X_em_mod = torch.tensor([b['t_min_of_day'] for b in batch], dtype=torch.int).unsqueeze(-1)
    X_em = torch.concat([X_em_mod, X_em_dow], axis=1)
    X_ct = torch.zeros((len(batch), len(GPS_FEATS)*2))
    for i,col in enumerate(GPS_FEATS):
        X_ct[:,i] = torch.tensor([b[col][0] for b in batch], dtype=torch.float)
        X_ct[:,i+len(GPS_FEATS)] = torch.tensor([b[col][-1] for b in batch], dtype=torch.float)
    return (X_em, X_ct), (y, y_no_norm)


def collate_static(batch):
    y = torch.tensor([b['cumul_time_s'][-1] for b in batch], dtype=torch.float)
    y_no_norm = torch.tensor([b['cumul_time_s_no_norm'][-1] for b in batch], dtype=torch.float)
    X_em_dow = torch.tensor([b['t_day_of_week'] for b in batch], dtype=torch.int).unsqueeze(-1)
    X_em_mod = torch.tensor([b['t_min_of_day'] for b in batch], dtype=torch.int).unsqueeze(-1)
    X_em = torch.concat([X_em_mod, X_em_dow], axis=1)
    X_ct = torch.zeros((len(batch), (len(GPS_FEATS)+len(STATIC_FEATS))*2))
    for i,col in enumerate(GPS_FEATS+STATIC_FEATS):
        X_ct[:,i] = torch.tensor([b[col][0] for b in batch], dtype=torch.float)
        X_ct[:,i+len(GPS_FEATS)+len(STATIC_FEATS)] = torch.tensor([b[col][-1] for b in batch], dtype=torch.float)
    return (X_em, X_ct), (y, y_no_norm)


def collate_realtime(batch):
    y = torch.tensor([b['cumul_time_s'][-1] for b in batch], dtype=torch.float)
    y_no_norm = torch.tensor([b['cumul_time_s_no_norm'][-1] for b in batch], dtype=torch.float)
    X_em_dow = torch.tensor([b['t_day_of_week'] for b in batch], dtype=torch.int).unsqueeze(-1)
    X_em_mod = torch.tensor([b['t_min_of_day'] for b in batch], dtype=torch.int).unsqueeze(-1)
    X_em = torch.concat([X_em_mod, X_em_dow], axis=1)
    X_ct = torch.zeros((len(batch), (len(GPS_FEATS)+len(STATIC_FEATS))*2))
    for i,col in enumerate(GPS_FEATS+STATIC_FEATS):
        X_ct[:,i] = torch.tensor([b[col][0] for b in batch], dtype=torch.float)
        X_ct[:,i+len(GPS_FEATS)+len(STATIC_FEATS)] = torch.tensor([b[col][-1] for b in batch], dtype=torch.float)
    grid_channels = batch[0]['grid'].shape[1]
    grid_n = batch[0]['grid'].shape[2]
    X_gr = torch.zeros((len(batch), grid_channels, grid_n, 2))
    X_gr[:,:,:,0] = torch.concat([torch.tensor(b['grid'][0,:,:], dtype=torch.float).unsqueeze(0) for b in batch], dim=0)
    X_gr[:,:,:,1] = torch.concat([torch.tensor(b['grid'][-1,:,:], dtype=torch.float).unsqueeze(0) for b in batch], dim=0)
    return (X_em, X_ct, X_gr), (y, y_no_norm)


def collate_seq(batch):
    y = torch.nn.utils.rnn.pad_sequence([torch.tensor(b['calc_time_s'], dtype=torch.float) for b in batch])
    y_no_norm = torch.nn.utils.rnn.pad_sequence([torch.tensor(b['calc_time_s_no_norm'], dtype=torch.float) for b in batch])
    X_sl = torch.tensor([len(b['calc_time_s']) for b in batch], dtype=torch.int)
    X_em_dow = torch.tensor([b['t_day_of_week'] for b in batch], dtype=torch.int).unsqueeze(-1)
    X_em_mod = torch.tensor([b['t_min_of_day'] for b in batch], dtype=torch.int).unsqueeze(-1)
    X_em = torch.concat([X_em_mod, X_em_dow], axis=1)
    X_ct = torch.zeros((torch.max(X_sl), len(batch), len(GPS_FEATS)))
    for i,col in enumerate(GPS_FEATS):
        X_ct[:,:,i] = torch.nn.utils.rnn.pad_sequence([torch.tensor(b[col], dtype=torch.float) for b in batch])
    return (X_em, X_ct, X_sl), (y, y_no_norm)


def collate_seq_static(batch):
    y = torch.nn.utils.rnn.pad_sequence([torch.tensor(b['calc_time_s'], dtype=torch.float) for b in batch])
    y_no_norm = torch.nn.utils.rnn.pad_sequence([torch.tensor(b['calc_time_s_no_norm'], dtype=torch.float) for b in batch])
    X_sl = torch.tensor([len(b['calc_time_s']) for b in batch], dtype=torch.int)
    X_em_dow = torch.tensor([b['t_day_of_week'] for b in batch], dtype=torch.int).unsqueeze(-1)
    X_em_mod = torch.tensor([b['t_min_of_day'] for b in batch], dtype=torch.int).unsqueeze(-1)
    X_em = torch.concat([X_em_mod, X_em_dow], axis=1)
    X_ct = torch.zeros((torch.max(X_sl), len(batch), len(GPS_FEATS+STATIC_FEATS)))
    for i,col in enumerate(GPS_FEATS+STATIC_FEATS):
        X_ct[:,:,i] = torch.nn.utils.rnn.pad_sequence([torch.tensor(b[col], dtype=torch.float) for b in batch])
    return (X_em, X_ct, X_sl), (y, y_no_norm)


def collate_seq_realtime(batch):
    y = torch.nn.utils.rnn.pad_sequence([torch.tensor(b['calc_time_s'], dtype=torch.float) for b in batch])
    y_no_norm = torch.nn.utils.rnn.pad_sequence([torch.tensor(b['calc_time_s_no_norm'], dtype=torch.float) for b in batch])
    X_sl = torch.tensor([len(b['calc_time_s']) for b in batch], dtype=torch.int)
    X_em_dow = torch.tensor([b['t_day_of_week'] for b in batch], dtype=torch.int).unsqueeze(-1)
    X_em_mod = torch.tensor([b['t_min_of_day'] for b in batch], dtype=torch.int).unsqueeze(-1)
    X_em = torch.concat([X_em_mod, X_em_dow], axis=1)
    X_ct = torch.zeros((torch.max(X_sl), len(batch), len(GPS_FEATS+STATIC_FEATS)))
    for i,col in enumerate(GPS_FEATS+STATIC_FEATS):
        X_ct[:,:,i] = torch.nn.utils.rnn.pad_sequence([torch.tensor(b[col], dtype=torch.float) for b in batch])
    X_gr = torch.nn.utils.rnn.pad_sequence([torch.tensor(b['grid'], dtype=torch.float) for b in batch], batch_first=True)
    X_gr = torch.swapaxes(X_gr, 1, 3)
    X_gr = torch.swapaxes(X_gr, 1, 2)
    return (X_em, X_ct, X_sl, X_gr), (y, y_no_norm)


def collate_deeptte(batch):
    stat_attrs = ['cumul_dist_m', 'cumul_time_s']
    info_attrs = ['t_day_of_week', 't_min_of_day']
    traj_attrs = GPS_FEATS
    attr, traj = {}, {}
    X_sl = torch.tensor([len(b['calc_time_s']) for b in batch], dtype=torch.int)
    traj['X_sl'] = X_sl
    traj['calc_time_s'] = torch.nn.utils.rnn.pad_sequence([torch.tensor(b['calc_time_s'], dtype=torch.float) for b in batch])
    labels = torch.tensor([b['cumul_time_s_no_norm'][-1] for b in batch], dtype=torch.float)
    for k in stat_attrs:
        attr[k] = torch.tensor([b[k][-1] for b in batch], dtype=torch.float)
    for k in info_attrs:
        attr[k] = torch.tensor([b[k] for b in batch], dtype=torch.int)
    for k in traj_attrs:
        traj[k] = torch.nn.utils.rnn.pad_sequence([torch.tensor(b[k], dtype=torch.float) for b in batch])
    return (attr, traj, labels)


def collate_deeptte_static(batch):
    stat_attrs = ['cumul_dist_m', 'cumul_time_s']
    info_attrs = ['t_day_of_week', 't_min_of_day']
    traj_attrs = GPS_FEATS+STATIC_FEATS
    attr, traj = {}, {}
    X_sl = torch.tensor([len(b['calc_time_s']) for b in batch], dtype=torch.int)
    traj['X_sl'] = X_sl
    traj['calc_time_s'] = torch.nn.utils.rnn.pad_sequence([torch.tensor(b['calc_time_s'], dtype=torch.float) for b in batch])
    labels = torch.tensor([b['cumul_time_s_no_norm'][-1] for b in batch], dtype=torch.float)
    for k in stat_attrs:
        attr[k] = torch.tensor([b[k][-1] for b in batch], dtype=torch.float)
    for k in info_attrs:
        attr[k] = torch.tensor([b[k] for b in batch], dtype=torch.int)
    for k in traj_attrs:
        traj[k] = torch.nn.utils.rnn.pad_sequence([torch.tensor(b[k], dtype=torch.float) for b in batch])
    return (attr, traj, labels)