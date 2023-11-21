import gc
import pickle

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
]
MISC_CAT_FEATS = [
    "file",
    "data_folder",
]
HOLDOUT_ROUTES = [
    100252,
    100139,
    102581,
    100341,
    102720,
    "ATB:Line:2_28",
    "ATB:Line:2_3",
    "ATB:Line:2_9",
    "ATB:Line:2_340",
    "ATB:Line:2_299",
]


def normalize(x, config_entry):
    mean = config_entry[0]
    std = config_entry[1]
    return (x - mean) / std


def denormalize(x, config_entry):
    mean = config_entry[0]
    std = config_entry[1]
    return x * std + mean


def create_config(data_lookup, idx):
    data = [data_lookup[x] for x in idx]
    config = {}
    for i,col in enumerate(LABEL_FEATS+EMBED_FEATS+GPS_FEATS+STATIC_FEATS+DEEPTTE_FEATS+MISC_CON_FEATS+MISC_CAT_FEATS):
        if col in LABEL_FEATS+GPS_FEATS+STATIC_FEATS+DEEPTTE_FEATS+MISC_CON_FEATS:
            col_data = np.concatenate([samp[:,i] for samp in data])
            col_mean = np.mean(col_data)
            col_sd = np.std(col_data)
            config[col] = (col_mean, col_sd)
    return config


class DictDataset(Dataset):
    """Load all data into memory as dataframe, provide samples by indexing groups."""
    def __init__(self, data_folders, dates, holdout_type=None, only_holdout=False, **kwargs):
        self.data_folder_codes = {}
        self.file_codes = {}
        self.grids = {}
        self.data = {}
        last_key = 0
        for n_df, data_folder in enumerate(data_folders):
            self.data_folder_codes[n_df] = data_folder
            self.file_codes[n_df] = {}
            self.grids[n_df] = {}
            for n_f, day in enumerate(dates):
                self.file_codes[n_df][n_f] = day.split('.')[0]
                d = pd.read_pickle(f"{data_folder}{day}")
                d['data_folder'] = n_df
                d['file'] = n_f
                d['cumul_dist_m'] = d['cumul_dist_km'] * 1000
                if holdout_type=='create':
                    self.holdout_routes = HOLDOUT_ROUTES
                    if only_holdout:
                        d = d[d['route_id'].isin(self.holdout_routes)]
                    else:
                        d = d[~d['route_id'].isin(self.holdout_routes)]
                elif holdout_type=='specify':
                    self.holdout_routes = kwargs['holdout_routes']
                    if only_holdout:
                        d = d[d['route_id'].isin(self.holdout_routes)]
                    else:
                        d = d[~d['route_id'].isin(self.holdout_routes)]
                d['shingle_id'] = d.groupby('shingle_id').ngroup()
                d = d.set_index('shingle_id')
                d.index = d.index + last_key
                self.data_colnames = LABEL_FEATS+EMBED_FEATS+GPS_FEATS+STATIC_FEATS+DEEPTTE_FEATS+MISC_CON_FEATS+MISC_CAT_FEATS
                d = d[self.data_colnames]
                x = {key: group.to_numpy().astype('int32') for key, group in d.groupby('shingle_id')}
                self.data.update(x)
                with open(f"{data_folder}/grid/{day}", 'rb') as f:
                    self.grids[n_df][n_f] = pickle.load(f)
                last_key = sorted(self.data)[-1] + 1
        self.config = None
        self.include_grid = False
    def __getitem__(self, index):
        sample_data = self.data[index]
        sample = {}
        for i,k in enumerate(LABEL_FEATS):
            sample[f"{k}_no_norm"] = sample_data[:,self.data_colnames.index(k)].astype(float)
        for i,k in enumerate(LABEL_FEATS+GPS_FEATS+STATIC_FEATS+DEEPTTE_FEATS+MISC_CON_FEATS):
            sample[k] = normalize(sample_data[:,self.data_colnames.index(k)].astype(float), self.config[k])
        for k in EMBED_FEATS:
            sample[k] = sample_data[:,self.data_colnames.index(k)].astype(int)[0]
        if self.include_grid:
            sample_file = sample_data[:,self.data_colnames.index('file')][0]
            sample_folder = sample_data[:,self.data_colnames.index('data_folder')][0]
            feat_idxs = (self.data_colnames.index('x'), self.data_colnames.index('y'), self.data_colnames.index('locationtime'))
            sample['grid'] = self.grids[sample_folder][sample_file].get_recent_points(sample_data[:,feat_idxs], 4)
        return sample
    def __len__(self):
        return len(self.data)
    def create_config(self, idxs):
        data = [self.data[x] for x in idxs]
        config = {}
        for i,col in enumerate(self.data_colnames):
            col_data = np.concatenate([samp[:,i] for samp in data])
            col_mean = np.mean(col_data)
            col_sd = np.std(col_data)
            config[col] = (col_mean, col_sd)
        return config


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
    stat_attrs = ['cumul_dist_km', 'cumul_time_s']
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
    stat_attrs = ['cumul_dist_km', 'cumul_time_s']
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