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
DEEPTTE_FEATS = [
    "cumul_dist_km",
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
    for col in LABEL_FEATS+GPS_FEATS+STATIC_FEATS+REALTIME_FEATS+DEEPTTE_FEATS:
        col_mean = np.mean(data[col].to_numpy())
        col_sd = np.std(data[col].to_numpy())
        config[col] = (col_mean, col_sd)
    return config


class ContentDataset(Dataset):
    """Load all data into memory as dataframe, provide samples by indexing groups."""
    def __init__(self, data_folders, dates, holdout_type=None, only_holdout=False, **kwargs):
        data = []
        for data_folder in data_folders:
            for day in dates:
                data.append(pd.read_pickle(f"{data_folder}{day}").to_crs('EPSG:4326'))
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
        self.feat_data = self.data[LABEL_FEATS+EMBED_FEATS+GPS_FEATS+STATIC_FEATS+REALTIME_FEATS+DEEPTTE_FEATS]
        self.groupdata = self.feat_data.groupby('shingle_id')
        self.config = None
    def __getitem__(self, index):
        sample_df = self.groupdata.get_group(index)
        sample = {}
        for k in LABEL_FEATS+GPS_FEATS+STATIC_FEATS+REALTIME_FEATS+DEEPTTE_FEATS:
            sample[k] = normalize(sample_df[k].to_numpy(), self.config[k])
        for k in EMBED_FEATS:
            sample[k] = sample_df[k].to_numpy()[0]
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


# def collate_realtime(batch):
#     y_col = "time_cumulative_s"
#     em_cols = ["timeID","weekID"]
#     first_cols = ["x_cent","y_cent","scheduled_time_s","stop_dist_km","passed_stops_n","speed_m_s","bearing"]
#     last_cols = ["x_cent","y_cent","scheduled_time_s","stop_dist_km","passed_stops_n","dist_cumulative_km","bearing"]
#     y = torch.tensor([b['norm_label'] for b in batch], dtype=torch.float)
#     batch_ct = [torch.tensor(b['samp'], dtype=torch.float) for b in batch]
#     first = torch.cat([b[0,:].unsqueeze(0) for b in batch_ct], axis=0)
#     last = torch.cat([b[-1,:].unsqueeze(0) for b in batch_ct], axis=0)
#     X_em = first[:,([FEATURE_COLS.index(z) for z in em_cols])].int()
#     X_ct_first = first[:,([FEATURE_COLS.index(z) for z in first_cols])]
#     X_ct_last = last[:,([FEATURE_COLS.index(z) for z in last_cols])]
#     X_ct = torch.cat([X_ct_first, X_ct_last], axis=1)
#     # Get speed/bearing/obs age from grid results; average out all values across timesteps; 1 value per cell/obs/variable
#     X_gr = torch.cat([torch.nanmean(torch.tensor(z['grid'], dtype=torch.float)[:,3:,:], dim=0).unsqueeze(0) for z in batch])
#     # X_gr = torch.cat([torch.nanmean(torch.tensor(z['grid'], dtype=torch.float)[:,3:,:,:,:], dim=0).unsqueeze(0) for z in batch])
#     # Replace any nans with the average for that feature
#     means = torch.nanmean(torch.swapaxes(X_gr, 0, 1).flatten(1), dim=1)
#     for i,m in enumerate(means):
#         X_gr[:,i,:] = torch.nan_to_num(X_gr[:,i,:], m)
#         # X_gr[:,i,:,:,:] = torch.nan_to_num(X_gr[:,i,:,:,:], m)
#     return [X_em, X_ct, X_gr], y


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


# def collate_seq_realtime(batch):
#     y_col = "time_calc_s"
#     em_cols = ["timeID","weekID"]
#     ct_cols = ["x_cent","y_cent","scheduled_time_s","stop_dist_km","stop_x_cent","stop_y_cent","passed_stops_n","bearing","dist_calc_km","dist_calc_km"]
#     y = torch.nn.utils.rnn.pad_sequence([torch.tensor(b['norm_label_seq'], dtype=torch.float) for b in batch], batch_first=True)
#     batch_ct = [torch.tensor(b['samp'], dtype=torch.float) for b in batch]
#     batch_gr = [torch.tensor(b['grid'], dtype=torch.float) for b in batch]
#     X_sl = torch.tensor([len(b) for b in batch_ct], dtype=torch.int)
#     first = torch.cat([b[0,:].unsqueeze(0) for b in batch_ct], axis=0)
#     X_em = first[:,([FEATURE_COLS.index(z) for z in em_cols])].int()
#     batch_ct = torch.nn.utils.rnn.pad_sequence(batch_ct, batch_first=True)
#     X_ct = batch_ct[:,:,([FEATURE_COLS.index(z) for z in ct_cols])]
#     # Get speed/bearing/obs age from grid results; average out all values across timesteps; 1 value per cell/obs/variable
#     X_gr = [b[:,3:,:] for b in batch_gr]
#     # X_gr = [b[:,3:,:,:,:] for b in batch_gr]
#     X_gr = torch.nn.utils.rnn.pad_sequence(X_gr, batch_first=True)
#     # Replace any nans with the average for that feature
#     means = torch.nanmean(torch.swapaxes(X_gr, 0, 2).flatten(1), dim=1)
#     for i,m in enumerate(means):
#         X_gr[:,:,i,:] = torch.nan_to_num(X_gr[:,:,i,:], m)
#         # X_gr[:,:,i,:,:,:] = torch.nan_to_num(X_gr[:,:,i,:,:,:], m)
#     return [X_em, X_ct, X_gr, X_sl], y


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


# def collate_deeptte_static(batch):
#     stat_attrs = ['dist_cumulative_km', 'time_cumulative_s']
#     stat_names = ['dist','time']
#     info_attrs = ['weekID', 'timeID']
#     traj_attrs = ['y_cent', 'x_cent', 'time_calc_s', 'dist_calc_km']
#     attr, traj = {}, {}
#     batch_ct = [torch.tensor(b['samp']) for b in data]
#     lens = np.array([len(b) for b in batch_ct])
#     for n,key in zip(stat_names, stat_attrs):
#         attr[n] = torch.FloatTensor([d['samp'][-1,FEATURE_COLS.index(key)] for d in data])
#     for key in info_attrs:
#         attr[key] = torch.LongTensor([int(d['samp'][0,FEATURE_COLS.index(key)]) for d in data])
#     for key in traj_attrs:
#         seqs = [d['samp'][:,FEATURE_COLS.index(key)] for d in data]
#         seqs = np.asarray(seqs, dtype=object)
#         mask = np.arange(lens.max()) < lens[:, None]
#         padded = np.zeros(mask.shape, dtype=np.float32)
#         padded[mask] = np.concatenate(seqs)
#         padded = torch.from_numpy(padded).float()
#         traj[key] = padded
#     lens = lens.tolist()
#     traj['lens'] = lens
#     return [attr, traj]