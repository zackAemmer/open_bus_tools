import torch
import torch.nn as nn
import torch.nn.functional as F

from openbustools.traveltime import utils


class Net(nn.Module):
    def __init__(self, kernel_size, num_filter):
        super(Net, self).__init__()

        self.kernel_size = kernel_size
        self.num_filter = num_filter

        self.build()

    def build(self):
        # self.process_coords = nn.Linear(4, 16) ### Change dim for remove states
        self.process_coords = nn.Linear(2, 16)
        self.conv = nn.Conv1d(16, self.num_filter, self.kernel_size)

    def forward(self, traj, config):
        lon = torch.unsqueeze(traj['y_cent'], dim = 2)
        lat = torch.unsqueeze(traj['x_cent'], dim = 2)

        locs = torch.cat((lon, lat), dim = 2)   ### Remove states 

        # map the coords into 16-dim vector
        locs = torch.tanh(self.process_coords(locs))   ### size [batch, max len of trajectory in batch, 16]
        locs = locs.permute(0, 2, 1)  

        conv_locs = F.elu(self.conv(locs))   ### size [batch, num_filter, max len of batch traj - kernel_size + 1]
        conv_locs = conv_locs.permute(0, 2, 1)   ### size [batch, max len - kernel_size + 1, num_filter]

        # calculate the dist for local paths
        local_dist = deeptte_utils.get_local_seq(traj['dist_calc_km'], self.kernel_size, config['dist_calc_km_mean'], config['dist_calc_km_std'])
        local_dist = torch.unsqueeze(local_dist, dim = 2)   ### ### size [batch, max len - kernel_size + 1, 1]

        conv_locs = torch.cat((conv_locs, local_dist), dim = 2)   ### size [batch, max len - kernel_size + 1, num_filter + 1]

        return conv_locs

