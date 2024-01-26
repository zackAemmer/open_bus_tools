from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from openbustools import trackcleaning


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
TRAIN_COLS = LABEL_FEATS+GPS_FEATS+STATIC_FEATS+DEEPTTE_FEATS
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
    """
    Create a configuration dictionary of feature mean and std based on the given samples.

    Args:
        samples (numpy.ndarray): The input samples to calculate from.

    Returns:
        dict: The configuration dictionary with statistics for each feature column.
    """
    config = {}
    config['sample_count'] = samples.shape[0]
    for i, col in enumerate(NUM_FEAT_COLS):
        config[col] = (np.mean(samples[:,i]), np.std(samples[:,i]))
    return config


class trajectoryDataset(Dataset):
    """
    A dataset class for handling trajectory inference.

    Args:
        traj (Trajectory): The trajectory object containing the track to infer.
        config (dict): Configuration settings for the dataset.

    Attributes:
        sample_lookup (dict): A dictionary mapping sample indices to data.
        config (dict): Configuration settings for the dataset.

    Methods:
        find_sample(index): Returns the sample data for the given index.
        __getitem__(index): Returns the normalized sample data for the given index.
        __len__(): Returns the number of samples in the dataset.
    """
    def __init__(self, traj, config):
        gdf = traj.gdf.copy()
        gdf['t_min_of_day'] = traj.traj_attr['t_min_of_day']
        gdf['t_day_of_week'] = traj.traj_attr['t_day_of_week']
        # Fill modeling features with -1 if not added to trajectory gdf
        gdf['shingle_id'] = -1
        gdf['route_id'] = -1
        for col in NUM_FEAT_COLS:
            if col not in gdf.columns:
                gdf[col] = -1
        _, data_n, _ = trackcleaning.extract_training_features(gdf)
        self.sample_lookup = {0: data_n}
        self.config = config
    def find_sample(self, index):
        """
        Returns the sample data for the given index.

        Args:
            index (int): The index of the sample.

        Returns:
            ndarray: The sample data.
        """
        return self.sample_lookup[index]
    def __getitem__(self, index):
        """
        Returns the normalized sample data for the given index.

        Args:
            index (int): The index of the sample.

        Returns:
            ndarray: The normalized sample data.
        """
        # Load if not in memory, normalize non-embedding columns
        sample_data = self.find_sample(index)[:].astype(np.float32)
        for i, col in enumerate(TRAIN_COLS):
            feat_idx = NUM_FEAT_COLS.index(col)
            sample_data[:,feat_idx] = normalize(sample_data[:,feat_idx], self.config[col])
        return sample_data

    def __len__(self):
        """
        Returns the number of samples in the dataset.

        Returns:
            int: The number of samples.
        """
        return len(self.sample_lookup.keys())


class NumpyDataset(Dataset):
    """
    A PyTorch dataset for loading data from numpy files.

    Args:
        data_folders (list): List of folders containing the data.
        train_days (list): List of days to use for training.
        holdout_routes (list, optional): List of routes to hold out from training. Defaults to [].
        only_holdouts (bool, optional): Whether to only load data from holdout routes. Defaults to False.
        load_in_memory (bool, optional): Whether to load the data in memory. Defaults to False.
        include_grid (bool, optional): Whether to include grid features. Defaults to False.

    Attributes:
        data_folders (list): List of folders containing the data.
        train_days (list): List of days to use for training.
        holdout_routes (list): List of routes to hold out from training.
        only_holdouts (bool): Whether to only load data from holdout routes.
        load_in_memory (bool): Whether to load the data in memory.
        include_grid (bool): Whether to include grid features.
        sample_lookup (dict): Dictionary mapping sample index to file, start, and length.
        open_data_files (dict): Dictionary of open data files.
        config (dict): Configuration generated from random data samples.

    Methods:
        find_sample(index): Retrieves sample data from the specified file or array based on the given index.
        __getitem__(index): Returns the sample data at the given index processed for training.
        __len__(): Returns the total number of samples in the dataset.
    """
    def __init__(self, data_folders, train_days, holdout_routes=[], only_holdouts=False, load_in_memory=False, include_grid=False, config=None):
        self.data_folders = [Path(f, "training") for f in data_folders]
        self.train_days = train_days
        self.holdout_routes = holdout_routes
        self.only_holdouts = only_holdouts
        self.load_in_memory = load_in_memory
        self.include_grid = include_grid
        self.config = config
        # Scan files to create sample lookup
        sample_index = 0
        self.sample_lookup = {}
        self.open_data_n_files = {}
        self.open_data_g_files = {}
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
                    # Separate out the holdout routes
                    shingle_route_ids = np.load(day_folder / f"{day_folder.name}_c.npy").astype(str).flatten()
                    holdout_mask = np.isin(shingle_route_ids, self.holdout_routes)
                    is_holdout = holdout_mask[shingle_start_indices]
                    if self.only_holdouts:
                        shingle_start_indices = shingle_start_indices[is_holdout]
                    else:
                        shingle_start_indices = shingle_start_indices[~is_holdout]
                    # Save start point and len of each sample in its array
                    for (i, j) in zip(shingle_start_indices, shingle_lens):
                        # Lookup is (file, start, length)
                        self.sample_lookup[sample_index] = (str(day_folder), i, j)
                        sample_index += 1
                    # Store sample features as either list of files, or arrays loaded in memory
                    if self.load_in_memory:
                        data_n = np.load(day_folder / f"{day_folder.name}_n.npy")
                        self.open_data_n_files[str(day_folder)] = data_n
                        if self.include_grid:
                            data_g = np.load(day_folder / f"{day_folder.name}_g.npy")
                            self.open_data_g_files[str(day_folder)] = data_g
                    else:
                        data_n = np.load(day_folder / f"{day_folder.name}_n.npy", mmap_mode='c')
                        self.open_data_n_files[str(day_folder)] = data_n
                        if self.include_grid:
                            data_g = np.load(day_folder / f"{day_folder.name}_g.npy", mmap_mode='c')
                            self.open_data_g_files[str(day_folder)] = data_g
        # If config not passed, create one by sampling random data
        if self.config is None:
            random_indices = np.random.choice(list(self.sample_lookup.keys()), max(1, int(len(self)*.10)), replace=False)
            config_samples = np.concatenate([self.find_sample(i) for i in random_indices])
            self.config = create_config(config_samples)
    def find_sample(self, index):
        """
        Returns the sample data for the given index.

        Args:
            index (int): The index of the sample.

        Returns:
            ndarray: The sample data.
        """
        # Lookup rows for sample from array or open file
        day_folder, shingle_start, shingle_length = self.sample_lookup[index]
        sample_data = self.open_data_n_files[day_folder][shingle_start:shingle_start+shingle_length,:]
        # Concatenate grid to end of sample features if included
        if self.include_grid:
            grid_data = self.open_data_g_files[day_folder][shingle_start:shingle_start+shingle_length,:]
            grid_data = np.reshape(grid_data, (grid_data.shape[0], grid_data.shape[1] * grid_data.shape[2]))
            sample_data = np.concatenate([sample_data, grid_data], axis=1)
        return sample_data
    def __getitem__(self, index):
        """
        Returns the normalized sample data for the given index.

        Args:
            index (int): The index of the sample.

        Returns:
            ndarray: The normalized sample data.
        """
        # Load if not in memory, normalize non-embedding columns
        sample_data = self.find_sample(index)[:].astype(np.float32)
        for i, col in enumerate(TRAIN_COLS):
            feat_idx = NUM_FEAT_COLS.index(col)
            sample_data[:,feat_idx] = normalize(sample_data[:,feat_idx], self.config[col])
        return sample_data
    def __len__(self):
        """
        Returns the number of samples in the dataset.

        Returns:
            int: The number of samples.
        """
        return len(self.sample_lookup.keys())


def collate_seq(batch):
    padded_batch = torch.nn.utils.rnn.pad_sequence([torch.tensor(b, dtype=torch.float) for b in batch])
    # Get all embedding features; repeat first point through sequence
    X_em = padded_batch[:,:,[NUM_FEAT_COLS.index(x) for x in EMBED_FEATS]]
    X_em = X_em[0,:,:].unsqueeze(0).repeat(X_em.shape[0],1,1).int()
    # Get all continuous training features
    X_ct = padded_batch[:,:,[NUM_FEAT_COLS.index(x) for x in GPS_FEATS]]
    # Get sequence lengths
    X_sl = torch.tensor([len(b) for b in batch], dtype=torch.int)
    # Get labels
    Y = padded_batch[:,:,NUM_FEAT_COLS.index('calc_time_s')]
    Y_agg = torch.tensor([b[-1,NUM_FEAT_COLS.index('cumul_time_s')] for b in batch], dtype=torch.float)
    return ((X_em, X_ct), (Y, Y_agg), X_sl)


def collate_seq_static(batch):
    padded_batch = torch.nn.utils.rnn.pad_sequence([torch.tensor(b, dtype=torch.float) for b in batch])
    # Get all embedding features; repeat first point through sequence
    X_em = padded_batch[:,:,[NUM_FEAT_COLS.index(x) for x in EMBED_FEATS]]
    X_em = X_em[0,:,:].unsqueeze(0).repeat(X_em.shape[0],1,1).int()
    # Get all continuous training features
    X_ct = padded_batch[:,:,[NUM_FEAT_COLS.index(x) for x in GPS_FEATS+STATIC_FEATS]]
    # Get sequence lengths
    X_sl = torch.tensor([len(b) for b in batch], dtype=torch.int)
    # Get labels
    Y = padded_batch[:,:,NUM_FEAT_COLS.index('calc_time_s')]
    Y_agg = torch.tensor([b[-1,NUM_FEAT_COLS.index('cumul_time_s')] for b in batch], dtype=torch.float)
    return ((X_em, X_ct), (Y, Y_agg), X_sl)


def collate_seq_realtime(batch):
    padded_batch = torch.nn.utils.rnn.pad_sequence([torch.tensor(b, dtype=torch.float) for b in batch])
    # Get all embedding features; repeat first point through sequence
    X_em = padded_batch[:,:,[NUM_FEAT_COLS.index(x) for x in EMBED_FEATS]]
    X_em = X_em[0,:,:].unsqueeze(0).repeat(X_em.shape[0],1,1).int()
    # Get all continuous training features
    X_ct = padded_batch[:,:,[NUM_FEAT_COLS.index(x) for x in GPS_FEATS+STATIC_FEATS]]
    # Get sequence lengths
    X_sl = torch.tensor([len(b) for b in batch], dtype=torch.int)
    # Get labels
    Y = padded_batch[:,:,NUM_FEAT_COLS.index('calc_time_s')]
    Y_agg = torch.tensor([b[-1,NUM_FEAT_COLS.index('cumul_time_s')] for b in batch], dtype=torch.float)
    # Get grid features in (N x C x L)
    X_gr = padded_batch[:,:,len(NUM_FEAT_COLS):]
    X_gr = torch.swapaxes(X_gr, 0, 1)
    X_gr = torch.swapaxes(X_gr, 1, 2)
    return ((X_em, X_ct, X_gr), (Y, Y_agg), X_sl)


def collate_deeptte(batch):
    stat_attrs = ['cumul_dist_m', 'cumul_time_s']
    info_attrs = ['t_day_of_week', 't_min_of_day']
    traj_attrs = GPS_FEATS
    attr, traj = {}, {}
    padded_batch = torch.nn.utils.rnn.pad_sequence([torch.tensor(b, dtype=torch.float) for b in batch])
    # Get sequence lengths
    X_sl = torch.tensor([len(b) for b in batch], dtype=torch.int)
    traj['X_sl'] = X_sl
    # Get labels for individual points and full trajectory
    traj['calc_time_s'] = padded_batch[:,:,NUM_FEAT_COLS.index('calc_time_s')]
    labels = torch.tensor([b[-1,NUM_FEAT_COLS.index('cumul_time_s')] for b in batch], dtype=torch.float)
    # Get all continuous and embedding features
    for k in stat_attrs:
        attr[k] = torch.tensor([b[-1,NUM_FEAT_COLS.index(k)] for b in batch], dtype=torch.float)
    for k in info_attrs:
        attr[k] = torch.tensor([b[0,NUM_FEAT_COLS.index(k)] for b in batch], dtype=torch.long)
    for k in traj_attrs:
        traj[k] = padded_batch[:,:,[NUM_FEAT_COLS.index(k)]]
    return (attr, traj, labels)


def collate_deeptte_static(batch):
    stat_attrs = ['cumul_dist_m', 'cumul_time_s']
    info_attrs = ['t_day_of_week', 't_min_of_day']
    traj_attrs = GPS_FEATS+STATIC_FEATS
    attr, traj = {}, {}
    padded_batch = torch.nn.utils.rnn.pad_sequence([torch.tensor(b, dtype=torch.float) for b in batch])
    # Get sequence lengths
    X_sl = torch.tensor([len(b) for b in batch], dtype=torch.int)
    traj['X_sl'] = X_sl
    # Get labels for individual points and full trajectory
    traj['calc_time_s'] = padded_batch[:,:,NUM_FEAT_COLS.index('calc_time_s')]
    labels = torch.tensor([b[-1,NUM_FEAT_COLS.index('cumul_time_s')] for b in batch], dtype=torch.float)
    # Get all continuous and embedding features
    for k in stat_attrs:
        attr[k] = torch.tensor([b[-1,NUM_FEAT_COLS.index(k)] for b in batch], dtype=torch.float)
    for k in info_attrs:
        attr[k] = torch.tensor([b[0,NUM_FEAT_COLS.index(k)] for b in batch], dtype=torch.long)
    for k in traj_attrs:
        traj[k] = padded_batch[:,:,[NUM_FEAT_COLS.index(k)]]
    return (attr, traj, labels)