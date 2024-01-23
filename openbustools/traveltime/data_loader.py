from functools import reduce
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import Dataset


# Feature columns used at various points in training and testing
LABEL_FEATS = [
    "calc_time_s",
    "cumul_time_s",
]
GPS_FEATS = [
    "calc_dist_m",
    "calc_bear_d",
    "x_cent",
    "y_cent",
    "elev_m"
]
STATIC_FEATS = [
    "calc_stop_dist_m",
    "sch_time_s",
    "passed_stops_n",
    "cumul_passed_stops_n",
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
    return (x * std) + mean


def create_config(samples):
    config = {}
    config['sample_count'] = samples.shape[0]
    for i, col in enumerate(NUM_FEAT_COLS):
        config[col] = (np.mean(samples[:,i]), np.std(samples[:,i]))
    return config


# def load_h5(data_folders, dates, only_holdout=False, **kwargs):
#     """Preload all samples from an h5py file into memory, outside of dataset."""
#     data = {}
#     # Set holdout routes, variable names
#     if 'holdout_routes' in kwargs:
#         holdout_routes = kwargs['holdout_routes']
#     else:
#         holdout_routes = []
#     current_max_key = 0
#     # Load all of the folder/day data into one dictionary, reindexing along way
#     for data_folder in data_folders:
#         with h5py.File(f"{data_folder}samples.hdf5", 'r') as f:
#             for day in dates:
#                 try:
#                     sids, sidxs = np.unique(f[day]['shingle_ids'], return_index=True)
#                     feats_n = np.split(f[day]['feats_n'], sidxs[1:], axis=0)
#                     feats_g = np.split(f[day]['feats_g'], sidxs[1:], axis=0)
#                     feats_c = np.split(f[day]['feats_c'], sidxs[1:], axis=0)
#                     # Keep either all but, or only samples from holdout routes
#                     if len(holdout_routes)>0:
#                         is_not_holdout = np.array([x[0].astype(str)[0] not in holdout_routes for x in feats_c])
#                         if only_holdout:
#                             is_not_holdout = np.invert(is_not_holdout)
#                         sids = sids[is_not_holdout]
#                         feats_n = list(compress(feats_n, is_not_holdout))
#                         feats_g = list(compress(feats_g, is_not_holdout))
#                         feats_c = list(compress(feats_c, is_not_holdout))
#                     sids = np.arange(current_max_key, current_max_key+len(sids))
#                     sample = {fs: {'feats_n': fn,'feats_g': fg,'feats_c': fc} for fs,fn,fg,fc in zip(sids,feats_n,feats_g,feats_c)}
#                     data.update(sample)
#                     current_max_key = sorted(data.keys())[-1]+1
#                 except KeyError:
#                     print(f"Day not found: {day}")
#     # Add sample key to all data entries that has normalized data
#     if 'config' in kwargs:
#         config = kwargs['config']
#     else:
#         config = create_config(data, list(data.keys()))
#     normalize_samples(data, config)
#     return (data, holdout_routes, config)


class NumpyDataset(Dataset):
    # TODO: Holdout routes, grid, option to take config in constructor, normalize samples
    def __init__(self, data_folders, train_days, holdout_routes=[], load_in_memory=False, include_grid=False):
        self.data_folders = [Path(f, "training") for f in data_folders]
        self.train_days = train_days
        self.holdout_routes = holdout_routes
        self.load_in_memory = load_in_memory
        self.include_grid = include_grid
        # Scan files to create sample lookup
        sample_index = 0
        self.sample_lookup = {}
        self.open_data_files = {}
        for data_folder in self.data_folders:
            # Get all subdirectories that have data
            for day_folder in data_folder.glob("*"):
                # Train on specific days
                if day_folder.name in self.train_days:
                    data_file = day_folder / f"{day_folder.name}_sid.npy"
                    # Load shingle ids for the day to get start points and lengths
                    shingle_ids = np.load(data_file)
                    shingle_start_indices = np.where(np.diff(shingle_ids, prepend=np.nan))[0]
                    shingle_lens = np.diff(np.append(shingle_start_indices, len(shingle_ids)))
                    # Save start point and len of each sample in array
                    for (i, j) in zip(shingle_start_indices, shingle_lens):
                        # Lookup is (file, start, length)
                        self.sample_lookup[sample_index] = (str(day_folder), i, j)
                        sample_index += 1
                    # Keep training features file open for each day
                    data_n = np.load(day_folder / f"{day_folder.name}_n.npy", mmap_mode='c')
                    self.open_data_files[str(day_folder)] = data_n
        # Create a config by sampling, to use in normalizing the data
        random_indices = np.random.choice(list(self.sample_lookup.keys()), max(1, int(len(self)*.20)), replace=False)
        config_samples = np.concatenate([self.__getitem__(i) for i in random_indices])
        self.config = create_config(config_samples)
    def __getitem__(self, index):
        day_folder, shingle_start, shingle_length = self.sample_lookup[index]
        sample_data = self.open_data_files[day_folder][shingle_start:shingle_start+shingle_length,:]
        # sample_data = normalize_samples(sample_data, self.config)
        return sample_data[:]
    def __len__(self):
        return len(self.sample_lookup.keys())


def collate_seq(batch):
    norm = torch.nn.utils.rnn.pad_sequence([torch.tensor(b, dtype=torch.float) for b in batch])
    no_norm = torch.nn.utils.rnn.pad_sequence([torch.tensor(b, dtype=torch.long) for b in batch])
    X_em = no_norm[:,:,[NUM_FEAT_COLS.index(x) for x in EMBED_FEATS]]
    X_em = X_em[0,:,:].unsqueeze(0).repeat(X_em.shape[0],1,1)
    X_ct = norm[:,:,[NUM_FEAT_COLS.index(x) for x in GPS_FEATS]]
    X_sl = torch.tensor([len(b) for b in batch], dtype=torch.long)
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