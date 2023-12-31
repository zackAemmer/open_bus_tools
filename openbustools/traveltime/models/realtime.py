import torch
from torch import nn


class GridFeedForward(nn.Module):
    def __init__(self, input_size, compression_size, hidden_size):
        super(GridFeedForward, self).__init__()
        self.input_size = input_size
        self.compression_size = compression_size
        self.hidden_size = hidden_size
        self.norm = nn.BatchNorm1d(self.input_size)
        self.linear_relu_stack_grid = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.compression_size),
            nn.ReLU()
        )
    def forward(self, x):
        # N x C x L (norm over n grid samples and over sequence length)
        out = x.view(x.shape[0], x.shape[1]*x.shape[2], x.shape[3])
        out = self.norm(out)
        out = torch.swapaxes(out, 1, 2)
        out = self.linear_relu_stack_grid(out)
        out = torch.swapaxes(out, 0, 1)
        return out