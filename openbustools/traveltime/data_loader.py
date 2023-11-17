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
    "calc_dist_km",
    "calc_bear_d",
    "x_cent",
    "y_cent",
]
STATIC_FEATS = [
    "calc_stop_dist_km",
    "sch_time_s",
    "pass_stops_n",
    "cumul_pass_stops_n",
]
DEEPTTE_FEATS = [
    "cumul_dist_km",
]
MISC_CON_FEATS = [
    "calc_speed_m_s"
]
MISC_CAT_FEATS = [
    "file",
    "data_folder",
    "x",
    "y",
    "locationtime"
]


def normalize(x, config_entry):
    mean = config_entry[0]
    std = config_entry[1]
    return (x - mean) / std


def denormalize(x, config_entry):
    mean = config_entry[0]
    std = config_entry[1]
    return x * std + mean


def create_config(data):
    config = {}
    for col in LABEL_FEATS+GPS_FEATS+STATIC_FEATS+DEEPTTE_FEATS+MISC_CON_FEATS:
        col_mean = np.mean(data[col].to_numpy())
        col_sd = np.std(data[col].to_numpy())
        config[col] = (col_mean, col_sd)
    return config


class DictDataset(Dataset):
    """Load all data into memory as dataframe, provide samples by indexing groups."""
    def __init__(self, data_folders, dates, holdout_type=None, only_holdout=False, **kwargs):
        data = []
        self.grids = {}
        for data_folder in data_folders:
            self.grids[data_folder] = {}
            for day in dates:
                d = pd.read_pickle(f"{data_folder}{day}").to_crs('EPSG:4326')
                d['data_folder'] = data_folder
                data.append(d)
                with open(f"{data_folder}/grid/{day}", 'rb') as f:
                    self.grids[data_folder][day.split('.')[0]] = pickle.load(f)
        data = pd.concat(data)
        # Deal with different scenarios for holdout route training/testing
        if holdout_type=='create':
            unique_routes = pd.unique(data['route_id'])
            self.holdout_routes = np.random.choice(pd.unique(data['route_id']), int(len(unique_routes)*.05))
            if only_holdout:
                data = data[data['route_id'].isin(self.holdout_routes)]
            else:
                data = data[~data['route_id'].isin(self.holdout_routes)]
        elif holdout_type=='specify':
            self.holdout_routes = kwargs['holdout_routes']
            if only_holdout:
                data = data[data['route_id'].isin(self.holdout_routes)]
            else:
                data = data[~data['route_id'].isin(self.holdout_routes)]
        data['shingle_id'] = data.groupby(['file','shingle_id']).ngroup()
        data = data.set_index('shingle_id')
        self.data = data
        self.feat_data = self.data[LABEL_FEATS+EMBED_FEATS+GPS_FEATS+STATIC_FEATS+DEEPTTE_FEATS+MISC_CON_FEATS+MISC_CAT_FEATS]
        self.data_lookup = self.feat_data.groupby('shingle_id').apply(lambda x: x.to_numpy()).to_dict()
        self.config = None
        self.include_grid = False
    def __getitem__(self, index):
        sample_df = self.data_lookup[index]
        sample = {}
        col_idxs = LABEL_FEATS+EMBED_FEATS+GPS_FEATS+STATIC_FEATS+DEEPTTE_FEATS+MISC_CON_FEATS+MISC_CAT_FEATS
        for i,k in enumerate(LABEL_FEATS+GPS_FEATS+STATIC_FEATS+DEEPTTE_FEATS+MISC_CON_FEATS):
            sample[k] = normalize(sample_df[:,col_idxs.index(k)].astype(float), self.config[k])
        for k in EMBED_FEATS:
            sample[k] = sample_df[:,col_idxs.index(k)].astype(int)[0]
        if self.include_grid:
            sample_file = sample_df[:,col_idxs.index('file')][0]
            sample_folder = sample_df[:,col_idxs.index('data_folder')][0]
            feat_idxs = (col_idxs.index('x'), col_idxs.index('y'), col_idxs.index('locationtime'))
            sample['grid'] = self.grids[sample_folder][sample_file].get_recent_points(sample_df[:,feat_idxs], 4)
        return sample
    def __len__(self):
        return np.max(self.data.index.to_numpy())


def collate(batch):
    y = torch.tensor([b['cumul_time_s'][-1] for b in batch], dtype=torch.float)
    X_em_dow = torch.tensor([b['t_day_of_week'] for b in batch], dtype=torch.int).unsqueeze(-1)
    X_em_mod = torch.tensor([b['t_min_of_day'] for b in batch], dtype=torch.int).unsqueeze(-1)
    X_em = torch.concat([X_em_mod, X_em_dow], axis=1)
    X_ct = torch.zeros((len(batch), len(GPS_FEATS)*2))
    for i,col in enumerate(GPS_FEATS):
        X_ct[:,i] = torch.tensor([b[col][0] for b in batch], dtype=torch.float)
        X_ct[:,i+len(GPS_FEATS)] = torch.tensor([b[col][-1] for b in batch], dtype=torch.float)
    return (X_em, X_ct), y


def collate_static(batch):
    y = torch.tensor([b['cumul_time_s'][-1] for b in batch], dtype=torch.float)
    X_em_dow = torch.tensor([b['t_day_of_week'] for b in batch], dtype=torch.int).unsqueeze(-1)
    X_em_mod = torch.tensor([b['t_min_of_day'] for b in batch], dtype=torch.int).unsqueeze(-1)
    X_em = torch.concat([X_em_mod, X_em_dow], axis=1)
    X_ct = torch.zeros((len(batch), (len(GPS_FEATS)+len(STATIC_FEATS))*2))
    for i,col in enumerate(GPS_FEATS+STATIC_FEATS):
        X_ct[:,i] = torch.tensor([b[col][0] for b in batch], dtype=torch.float)
        X_ct[:,i+len(GPS_FEATS)+len(STATIC_FEATS)] = torch.tensor([b[col][-1] for b in batch], dtype=torch.float)
    return (X_em, X_ct), y


def collate_realtime(batch):
    y = torch.tensor([b['cumul_time_s'][-1] for b in batch], dtype=torch.float)
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
    return (X_em, X_ct, X_gr), y


def collate_seq(batch):
    y = torch.nn.utils.rnn.pad_sequence([torch.tensor(b['calc_time_s'], dtype=torch.float) for b in batch])
    X_sl = torch.tensor([len(b['calc_time_s']) for b in batch], dtype=torch.int)
    X_em_dow = torch.tensor([b['t_day_of_week'] for b in batch], dtype=torch.int).unsqueeze(-1)
    X_em_mod = torch.tensor([b['t_min_of_day'] for b in batch], dtype=torch.int).unsqueeze(-1)
    X_em = torch.concat([X_em_mod, X_em_dow], axis=1)
    X_ct = torch.zeros((torch.max(X_sl), len(batch), len(GPS_FEATS)))
    for i,col in enumerate(GPS_FEATS):
        X_ct[:,:,i] = torch.nn.utils.rnn.pad_sequence([torch.tensor(b[col], dtype=torch.float) for b in batch])
    return (X_em, X_ct, X_sl), y


def collate_seq_static(batch):
    y = torch.nn.utils.rnn.pad_sequence([torch.tensor(b['calc_time_s'], dtype=torch.float) for b in batch])
    X_sl = torch.tensor([len(b['calc_time_s']) for b in batch], dtype=torch.int)
    X_em_dow = torch.tensor([b['t_day_of_week'] for b in batch], dtype=torch.int).unsqueeze(-1)
    X_em_mod = torch.tensor([b['t_min_of_day'] for b in batch], dtype=torch.int).unsqueeze(-1)
    X_em = torch.concat([X_em_mod, X_em_dow], axis=1)
    X_ct = torch.zeros((torch.max(X_sl), len(batch), len(GPS_FEATS+STATIC_FEATS)))
    for i,col in enumerate(GPS_FEATS+STATIC_FEATS):
        X_ct[:,:,i] = torch.nn.utils.rnn.pad_sequence([torch.tensor(b[col], dtype=torch.float) for b in batch])
    return (X_em, X_ct, X_sl), y


def collate_seq_realtime(batch):
    y = torch.nn.utils.rnn.pad_sequence([torch.tensor(b['calc_time_s'], dtype=torch.float) for b in batch])
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
    return (X_em, X_ct, X_sl, X_gr), y


def collate_deeptte(batch):
    stat_attrs = ['cumul_dist_km', 'cumul_time_s']
    info_attrs = ['t_day_of_week', 't_min_of_day']
    traj_attrs = GPS_FEATS
    attr, traj = {}, {}
    X_sl = torch.tensor([len(b['calc_time_s']) for b in batch], dtype=torch.int)
    traj['X_sl'] = X_sl
    traj['calc_time_s'] = torch.nn.utils.rnn.pad_sequence([torch.tensor(b['calc_time_s'], dtype=torch.float) for b in batch])
    for k in stat_attrs:
        attr[k] = torch.tensor([b[k][-1] for b in batch], dtype=torch.float)
    for k in info_attrs:
        attr[k] = torch.tensor([b[k] for b in batch], dtype=torch.int)
    for k in traj_attrs:
        traj[k] = torch.nn.utils.rnn.pad_sequence([torch.tensor(b[k], dtype=torch.float) for b in batch])
    return (attr, traj)


def collate_deeptte_static(batch):
    stat_attrs = ['cumul_dist_km', 'cumul_time_s']
    info_attrs = ['t_day_of_week', 't_min_of_day']
    traj_attrs = GPS_FEATS+STATIC_FEATS
    attr, traj = {}, {}
    X_sl = torch.tensor([len(b['calc_time_s']) for b in batch], dtype=torch.int)
    traj['X_sl'] = X_sl
    traj['calc_time_s'] = torch.nn.utils.rnn.pad_sequence([torch.tensor(b['calc_time_s'], dtype=torch.float) for b in batch])
    for k in stat_attrs:
        attr[k] = torch.tensor([b[k][-1] for b in batch], dtype=torch.float)
    for k in info_attrs:
        attr[k] = torch.tensor([b[k] for b in batch], dtype=torch.int)
    for k in traj_attrs:
        traj[k] = torch.nn.utils.rnn.pad_sequence([torch.tensor(b[k], dtype=torch.float) for b in batch])
    return (attr, traj)