import torch
import torch.nn as nn
import torch.nn.functional as F

from openbustools.traveltime.models.deeptte import deeptte_utils


class Net(nn.Module):
    def __init__(self, kernel_size, num_filter):
        super(Net, self).__init__()
        self.kernel_size = kernel_size
        self.num_filter = num_filter
        self.build()
    def build(self):
        self.process_coords = nn.Linear(2, 16)
        self.conv = nn.Conv1d(16, self.num_filter, self.kernel_size, padding=1)
    def forward(self, traj):
        lon = torch.unsqueeze(traj['x_cent'], dim=2).squeeze(-1)
        lat = torch.unsqueeze(traj['y_cent'], dim=2).squeeze(-1)
        locs = torch.cat((lon, lat), dim=2)
        # Map the coords into 16-dim vector
        locs = torch.tanh(self.process_coords(locs))
        locs = torch.swapaxes(locs, 0, 1)
        locs = torch.swapaxes(locs, 1, 2)
        conv_locs = F.elu(self.conv(locs))
        conv_locs = torch.swapaxes(conv_locs, 0, 1)
        conv_locs = torch.swapaxes(conv_locs, 0, 2)
        # Calculate the dist for local paths
        local_dist = traj['calc_dist_m']
        conv_locs = torch.cat((conv_locs, local_dist), dim=2)
        return conv_locs