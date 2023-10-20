import time
import utils

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import numpy as np
import ujson as json

"""
Each JSON file contains a certain number of trips represented as a dict with the following keys:
Features associated with entire trip i.e. len(dict[key]) == 1 (categorical unless o/w stated)
1. driverID
2. weekID
3. dateID
5. timeID (start time of trip)
5. dist - continuous
6. time - continuous (ground truth, total travel time) 

Features associated with each ping i.e. len(dict[key]) >= 1(all continuous)
1. lat
2. lon
3. dist_gap
4. time_gap
5. states (optional)
"""


class MySet(Dataset):
    def __init__(self, input_file, network_folder, fold_num=None, n_folds=None, flag=None, keep_chunks=None, data_subset=None, holdout_routes=None, keep_only_holdout_routes=False):
        # Load data from file
        with open('./data/' + network_folder + "/" + input_file, 'r') as f:
            ### readLines() outputs a list of strings, each strings represents a dict with \n at the end
            self.content = f.readlines()
            ### json.loads() returns each string as a dict, content is now a map object i.e. list of dicts
            self.content = list(map(lambda x: json.loads(x), self.content))
            ### gets the number of trajectories in each trip, lengths is a map object i.e. list of int 
            self.lengths = list(map(lambda x: len(x['lon']), self.content))
        # Limit to data from this fold if training
        if keep_chunks=="train":
            n_per_fold = len(self.content) // n_folds
            mask = np.ones(len(self.content), bool)
            mask[fold_num*n_per_fold:(fold_num+1)*n_per_fold] = 0
            self.content = [item for item, keep in zip(self.content, mask) if keep]
            self.lengths = [item for item, keep in zip(self.lengths, mask) if keep]
        elif keep_chunks=="test":
            n_per_fold = len(self.content) // n_folds
            mask = np.ones(len(self.content), bool)
            mask[fold_num*n_per_fold:(fold_num+1)*n_per_fold] = 0
            self.content = [item for item, keep in zip(self.content, mask) if not keep]
            self.lengths = [item for item, keep in zip(self.lengths, mask) if not keep]
        # Else if keep_chunks == "EvalSet" or "KCM_KCM" etc. do nothing; keep all the data
        # Holdout routes for generalization
        if holdout_routes is not None:
            if keep_only_holdout_routes:
                keep_idx = [str(sample['route_id']) in holdout_routes for sample in self.content]
                self.content = [x for i,x in enumerate(self.content) if keep_idx[i]]
            else:
                keep_idx = [str(sample['route_id']) not in holdout_routes for sample in self.content]
                self.content = [x for i,x in enumerate(self.content) if keep_idx[i]]
        # Subset data for faster evaluation
        if data_subset is not None:
            if data_subset < 1:
                self.content = np.random.choice(self.content, int(data_subset*len(self.content)))
            else:
                self.content = np.random.choice(self.content, data_subset)

    def __getitem__(self, idx):
        return self.content[idx]

    def __len__(self):
        return len(self.content)

### This function is called when setting up the PyTorch dataloader
### Not quite sure how this works together with BatchSampler, but guess it is called on each batch
def collate_fn(data):
    stat_attrs = ['dist', 'time']
    info_attrs = ['weekID', 'timeID']
    traj_attrs = ['lon', 'lat', 'time_gap', 'dist_gap']
    # Add additional features
    traj_attrs = ['lon','lat','time_gap','dist_gap','bearing','scheduled_time_s','stop_dist_km','stop_x_cent','stop_y_cent','passed_stops_n']

    attr, traj = {}, {}

    ### item refers to each trip, len(item['lon']) would return length of each trip
    ### lens is an array of length of each trip in the batch
    lens = np.asarray([len(item['lon']) for item in data])

    ### Since these features are continuous, then normalise them
    for key in stat_attrs:
        x = torch.FloatTensor([item[key] for item in data])
        attr[key] = utils.normalize(x, key)

    for key in info_attrs:
        attr[key] = torch.LongTensor([item[key] for item in data])

    for key in traj_attrs:
        # pad to the max length
        ### Each element in seqs is a list of values for that variable
        seqs = np.asarray([item[key] for item in data], dtype=object)
        ### Creates a mask according to length of each trip wrt maximum trip length
        mask = np.arange(lens.max()) < lens[:, None]
        padded = np.zeros(mask.shape, dtype = np.float32)
        ### padded is a 2D array containing the padded sequence of values for each trip
        ### Each row represents a trip padded to the maximum length of trips in the batch
        ### Alternatively could use torch.nn.utils.rnn.pad_sequence here 
        padded[mask] = np.concatenate(seqs)

        if key in traj_attrs:
            padded = utils.normalize(padded, key)

        ### Convert to torch tensor
        padded = torch.from_numpy(padded).float()
        traj[key] = padded

    lens = lens.tolist()
    traj['lens'] = lens

    return attr, traj


### This function is evoked when setting up the PyTorch dataloader
class BatchSampler:
    def __init__(self, dataset, batch_size):
        self.count = len(dataset)
        self.batch_size = batch_size
        self.lengths = dataset.lengths
        self.indices = list(range(self.count))

    def __iter__(self):
        '''
        Divide the data into chunks with size = batch_size * 100
        sort by the length in one chunk
        '''
        np.random.shuffle(self.indices)

        chunk_size = self.batch_size * 100

        chunks = (self.count + chunk_size - 1) // chunk_size

        # re-arrange indices to minimize the padding
        ### Sorting is done here to prepare inputs for the temporal LSTM layer further down
        ### rnn.packed_padded_sequence with enforce_sorted=True is used in SpatioTemporal.py
        ### This requires the inputs to be sorted according to descending length, which is done here
        for i in range(chunks):
            partial_indices = self.indices[i * chunk_size: (i + 1) * chunk_size]
            partial_indices.sort(key = lambda x: self.lengths[x], reverse = True)
            self.indices[i * chunk_size: (i + 1) * chunk_size] = partial_indices

        # yield batch
        batches = (self.count - 1 + self.batch_size) // self.batch_size

        for i in range(batches):
            yield self.indices[i * self.batch_size: (i + 1) * self.batch_size]

    def __len__(self):
        return (self.count + self.batch_size - 1) // self.batch_size


def get_loader(input_file, network_folder, batch_size, fold_num=None, n_folds=None, flag=None, keep_chunks=None, data_subset=None, holdout_routes=None, keep_only_holdout_routes=False):
    dataset = MySet(input_file=input_file, network_folder=network_folder, fold_num=fold_num, n_folds=n_folds, flag=flag, keep_chunks=keep_chunks, data_subset=data_subset, holdout_routes=holdout_routes, keep_only_holdout_routes=keep_only_holdout_routes)
    ### dataset is a self-defined MySet object of training data with attributes content and lengths (of type map)
    batch_sampler = BatchSampler(dataset, batch_size)
    data_loader = DataLoader(dataset = dataset, \
                             batch_size = 1, \
                             collate_fn = collate_fn, \
                             num_workers = 4,
                             batch_sampler = batch_sampler,
                             pin_memory = True
    )
    return data_loader
