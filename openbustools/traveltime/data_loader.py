import json
import os
from random import sample

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
REALTIME_FEATS = [
    "calc_speed_m_s",
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
    for col in LABEL_FEATS+GPS_FEATS+STATIC_FEATS:
        col_mean = np.mean(data[col].to_numpy())
        col_sd = np.std(data[col].to_numpy())
        config[col] = (col_mean, col_sd)
    return config


class ContentDataset(Dataset):
    """Load all data into memory as dataframe, provide samples by indexing groups."""
    def __init__(self, data_folder, dates, holdout_type=None, only_holdout=False, **kwargs):
        data = []
        for day in dates:
            data.append(pd.read_pickle(f"{data_folder}{day}"))
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
        self.feat_data = self.data[LABEL_FEATS+EMBED_FEATS+GPS_FEATS+STATIC_FEATS]
        self.groupdata = self.feat_data.groupby('shingle_id')
        self.config = None
    def __getitem__(self, index):
        sample_df = self.groupdata.get_group(index)
        sample = {}
        for k in LABEL_FEATS+GPS_FEATS+STATIC_FEATS:
            sample[k] = normalize(sample_df[k].to_numpy(), self.config[k])
        for k in EMBED_FEATS:
            sample[k] = sample_df[k].to_numpy()[0]
        return sample
    def __len__(self):
        return np.max(self.data.index.to_numpy())


def collate_avg(batch):
    cols = ["speed_m_s","dist_calc_km","timeID_s","time_cumulative_s"]
    col_idxs = [FEATURE_COLS.index(cname) for cname in cols]
    avg_speeds = [np.mean(b['samp'][:,col_idxs[0]]) for b in batch]
    tot_dists = [np.sum(b['samp'][:,col_idxs[1]])*1000 for b in batch]
    start_times = [b['samp'][0,col_idxs[2]]//3600 for b in batch]
    tot_times = [b['samp'][-1,col_idxs[3]] for b in batch]
    return (avg_speeds, tot_dists, start_times, tot_times)


def collate_sch(batch):
    cols = ["scheduled_time_s","time_cumulative_s"]
    col_idxs = [FEATURE_COLS.index(cname) for cname in cols]
    sch_times = [b['samp'][-1,col_idxs[0]] for b in batch]
    tot_times = [b['samp'][-1,col_idxs[1]] for b in batch]
    return (sch_times, tot_times)


def collate_per(batch):
    seq_lens = [b['samp'].shape[0] for b in batch]
    cols = ["time_cumulative_s"]
    col_idxs = [FEATURE_COLS.index(cname) for cname in cols]
    tot_times = [b['samp'][-1,col_idxs[0]] for b in batch]
    return (seq_lens, tot_times)


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
    y_col = "time_cumulative_s"
    em_cols = ["timeID","weekID"]
    first_cols = ["x_cent","y_cent","scheduled_time_s","stop_dist_km","passed_stops_n","speed_m_s","bearing"]
    last_cols = ["x_cent","y_cent","scheduled_time_s","stop_dist_km","passed_stops_n","dist_cumulative_km","bearing"]
    y = torch.tensor([b['norm_label'] for b in batch], dtype=torch.float)
    batch_ct = [torch.tensor(b['samp'], dtype=torch.float) for b in batch]
    first = torch.cat([b[0,:].unsqueeze(0) for b in batch_ct], axis=0)
    last = torch.cat([b[-1,:].unsqueeze(0) for b in batch_ct], axis=0)
    X_em = first[:,([FEATURE_COLS.index(z) for z in em_cols])].int()
    X_ct_first = first[:,([FEATURE_COLS.index(z) for z in first_cols])]
    X_ct_last = last[:,([FEATURE_COLS.index(z) for z in last_cols])]
    X_ct = torch.cat([X_ct_first, X_ct_last], axis=1)
    # Get speed/bearing/obs age from grid results; average out all values across timesteps; 1 value per cell/obs/variable
    X_gr = torch.cat([torch.nanmean(torch.tensor(z['grid'], dtype=torch.float)[:,3:,:], dim=0).unsqueeze(0) for z in batch])
    # X_gr = torch.cat([torch.nanmean(torch.tensor(z['grid'], dtype=torch.float)[:,3:,:,:,:], dim=0).unsqueeze(0) for z in batch])
    # Replace any nans with the average for that feature
    means = torch.nanmean(torch.swapaxes(X_gr, 0, 1).flatten(1), dim=1)
    for i,m in enumerate(means):
        X_gr[:,i,:] = torch.nan_to_num(X_gr[:,i,:], m)
        # X_gr[:,i,:,:,:] = torch.nan_to_num(X_gr[:,i,:,:,:], m)
    return [X_em, X_ct, X_gr], y


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
    y_col = "time_calc_s"
    em_cols = ["timeID","weekID"]
    ct_cols = ["x_cent","y_cent","scheduled_time_s","stop_dist_km","stop_x_cent","stop_y_cent","passed_stops_n","bearing","dist_calc_km","dist_calc_km"]
    y = torch.nn.utils.rnn.pad_sequence([torch.tensor(b['norm_label_seq'], dtype=torch.float) for b in batch], batch_first=True)
    batch_ct = [torch.tensor(b['samp'], dtype=torch.float) for b in batch]
    batch_gr = [torch.tensor(b['grid'], dtype=torch.float) for b in batch]
    X_sl = torch.tensor([len(b) for b in batch_ct], dtype=torch.int)
    first = torch.cat([b[0,:].unsqueeze(0) for b in batch_ct], axis=0)
    X_em = first[:,([FEATURE_COLS.index(z) for z in em_cols])].int()
    batch_ct = torch.nn.utils.rnn.pad_sequence(batch_ct, batch_first=True)
    X_ct = batch_ct[:,:,([FEATURE_COLS.index(z) for z in ct_cols])]
    # Get speed/bearing/obs age from grid results; average out all values across timesteps; 1 value per cell/obs/variable
    X_gr = [b[:,3:,:] for b in batch_gr]
    # X_gr = [b[:,3:,:,:,:] for b in batch_gr]
    X_gr = torch.nn.utils.rnn.pad_sequence(X_gr, batch_first=True)
    # Replace any nans with the average for that feature
    means = torch.nanmean(torch.swapaxes(X_gr, 0, 2).flatten(1), dim=1)
    for i,m in enumerate(means):
        X_gr[:,:,i,:] = torch.nan_to_num(X_gr[:,:,i,:], m)
        # X_gr[:,:,i,:,:,:] = torch.nan_to_num(X_gr[:,:,i,:,:,:], m)
    return [X_em, X_ct, X_gr, X_sl], y


def collate_deeptte(data):
    stat_attrs = ['dist_cumulative_km', 'time_cumulative_s']
    stat_names = ['dist','time']
    info_attrs = ['weekID', 'timeID']
    traj_attrs = ['y_cent','x_cent','time_calc_s','dist_calc_km','bearing','scheduled_time_s','stop_dist_km','stop_x_cent','stop_y_cent','passed_stops_n']
    attr, traj = {}, {}
    batch_ct = [torch.tensor(b['samp']) for b in data]
    lens = np.array([len(b) for b in batch_ct])
    for n,key in zip(stat_names, stat_attrs):
        attr[n] = torch.FloatTensor([d['samp'][-1,FEATURE_COLS.index(key)] for d in data])
    for key in info_attrs:
        attr[key] = torch.LongTensor([int(d['samp'][0,FEATURE_COLS.index(key)]) for d in data])
    for key in traj_attrs:
        seqs = [d['samp'][:,FEATURE_COLS.index(key)] for d in data]
        seqs = np.asarray(seqs, dtype=object)
        mask = np.arange(lens.max()) < lens[:, None]
        padded = np.zeros(mask.shape, dtype=np.float32)
        padded[mask] = np.concatenate(seqs)
        padded = torch.from_numpy(padded).float()
        traj[key] = padded
    lens = lens.tolist()
    traj['lens'] = lens
    return [attr, traj]


def collate_deeptte_static(data):
    stat_attrs = ['dist_cumulative_km', 'time_cumulative_s']
    stat_names = ['dist','time']
    info_attrs = ['weekID', 'timeID']
    traj_attrs = ['y_cent', 'x_cent', 'time_calc_s', 'dist_calc_km']
    attr, traj = {}, {}
    batch_ct = [torch.tensor(b['samp']) for b in data]
    lens = np.array([len(b) for b in batch_ct])
    for n,key in zip(stat_names, stat_attrs):
        attr[n] = torch.FloatTensor([d['samp'][-1,FEATURE_COLS.index(key)] for d in data])
    for key in info_attrs:
        attr[key] = torch.LongTensor([int(d['samp'][0,FEATURE_COLS.index(key)]) for d in data])
    for key in traj_attrs:
        seqs = [d['samp'][:,FEATURE_COLS.index(key)] for d in data]
        seqs = np.asarray(seqs, dtype=object)
        mask = np.arange(lens.max()) < lens[:, None]
        padded = np.zeros(mask.shape, dtype=np.float32)
        padded[mask] = np.concatenate(seqs)
        padded = torch.from_numpy(padded).float()
        traj[key] = padded
    lens = lens.tolist()
    traj['lens'] = lens
    return [attr, traj]


# class LoadSliceDataset(Dataset):
#     def __init__(self, file_path, config, grid=None, holdout_routes=None, keep_only_holdout=False, add_grid_features=False, skip_gtfs=False):
#         self.file_path = file_path
#         self.config = config
#         self.grid = grid
#         self.holdout_routes = holdout_routes
#         self.keep_only_holdout = keep_only_holdout
#         self.add_grid_features = add_grid_features
#         self.skip_gtfs = skip_gtfs
#         # Necessary to convert from np array tabular format saved in h5 files
#         if not self.skip_gtfs:
#             self.col_names = FEATURE_COLS
#         else:
#             self.col_names = SKIP_FEATURE_COLS
#         # Cache column name indices
#         self.time_cumulative_s_idx = self.col_names.index("time_cumulative_s")
#         self.time_calc_s_idx = self.col_names.index("time_calc_s")
#         self.x_idx = self.col_names.index("x")
#         self.y_idx = self.col_names.index("y")
#         self.locationtime_idx = self.col_names.index("locationtime")
#         # Keep open files for the dataset
#         self.h5_lookup = {}
#         self.base_path = "/".join(self.file_path.split("/")[:-1])+"/"
#         self.train_or_test = self.file_path.split("/")[-1]
#         for filename in os.listdir(self.base_path):
#             if filename.startswith(self.train_or_test) and filename.endswith(".h5"):
#                 self.h5_lookup[filename] = h5py.File(f"{self.base_path}{filename}", 'r')['tabular_data']
#         # Read shingle lookup corresponding to dataset
#         # This is a list of keys that will be filtered and each point to a sample
#         with open(f"{self.file_path}_shingle_config.json") as f:
#             self.shingle_lookup = json.load(f)
#             self.shingle_keys = list(self.shingle_lookup.keys())
#         # Filter out (or keep exclusively) any routes that are used for generalization tests
#         if self.holdout_routes is not None:
#             holdout_idxs = [self.shingle_lookup[x]['route_id'] in self.holdout_routes for x in self.shingle_lookup]
#             if self.keep_only_holdout==True:
#                 self.shingle_keys = [i for (i,v) in zip(self.shingle_keys, holdout_idxs) if v]
#             else:
#                 self.shingle_keys = [i for (i,v) in zip(self.shingle_keys, holdout_idxs) if not v]
#     def __getitem__(self, index):
#         # Get information on shingle file location and lines; read specific shingle lines from specific shingle file
#         samp_dict = self.shingle_lookup[self.shingle_keys[index]]
#         samp = self.h5_lookup[f"{self.train_or_test}_data_{samp_dict['network']}_{samp_dict['file_num']}.h5"][samp_dict['start_idx']:samp_dict['end_idx']]
#         label = samp[-1,self.time_cumulative_s_idx]
#         norm_label = (label - self.config['time_mean']) / self.config['time_std']
#         label_seq = samp[:,self.time_calc_s_idx]
#         norm_label_seq = (label_seq - self.config['time_calc_s_mean']) / self.config['time_calc_s_std']
#         if not self.add_grid_features:
#             return {"samp": samp, "norm_label": norm_label, "norm_label_seq": norm_label_seq}
#         else:
#             xbin_idxs, ybin_idxs = self.grid.digitize_points(samp[:,self.x_idx], samp[:,self.y_idx])
#             grid_features = self.grid.get_recent_points(xbin_idxs, ybin_idxs, samp[:,self.locationtime_idx], 3)
#             return {"samp": samp, "grid": grid_features, "norm_label": norm_label, "norm_label_seq": norm_label_seq}
#     def __len__(self):
#         return len(self.shingle_keys)
#     def get_all_samples(self, keep_cols, indexes=None):
#         # Read all h5 files in run base directory; get all point obs
#         # Much faster way to get all points, but does not maintain unique samples
#         res = []
#         for k in list(self.h5_lookup.keys()):
#             df = self.h5_lookup[k][:]
#             df = pd.DataFrame(df, columns=self.col_names)
#             df['shingle_id'] = df['shingle_id'].astype(int)
#             df = df[keep_cols]
#             res.append(df)
#         res = pd.concat(res)
#         if indexes is not None:
#             # Indexes are in order, but shingle_id's are not; get shingle id for each keep index and filter
#             keep_shingles = [self.shingle_lookup[self.shingle_keys[i]]['shingle_id'] for i in indexes]
#             res = res[res['shingle_id'].isin(keep_shingles)]
#         return res
#     def get_all_samples_shingle_accurate(self, n_samples):
#         # Iterate every item in dataset
#         # Slower way to get data out, but keeps all samples unique
#         idxs = sample(list(np.arange(self.__len__())), n_samples)
#         z = []
#         for i in idxs:
#             z_data = pd.DataFrame(self.__getitem__(i)['samp'], columns=self.col_names)
#             z_dict = self.shingle_lookup[str(i)]
#             z_data['file'] = z_dict['file']
#             z_data['trip_id'] = z_dict['trip_id']
#             z_data['route_id'] = z_dict['route_id']
#             z_data['file_num'] = z_dict['file_num']
#             z.append(z_data)
#         res = pd.concat(z)
#         res['stop_x'] = res['stop_x_cent'] + self.config['coord_ref_center'][0][0]
#         res['stop_y'] = res['stop_y_cent'] + self.config['coord_ref_center'][0][1]
#         return res