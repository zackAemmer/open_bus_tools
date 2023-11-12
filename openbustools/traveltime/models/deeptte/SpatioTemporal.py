import torch
from torch.autograd import Variable
import torch.nn as nn

from openbustools.traveltime.models.deeptte import GeoConv


class Net(nn.Module):
    def __init__(self, attr_size, kernel_size=3, num_filter=32, pooling_method='attention', rnn='lstm'):
        super(Net, self).__init__()
        self.kernel_size=kernel_size
        self.num_filter=num_filter
        self.pooling_method=pooling_method
        self.geo_conv=GeoConv.Net(kernel_size=kernel_size, num_filter=num_filter)
        if rnn=='lstm':
            self.rnn = nn.LSTM(
                input_size=num_filter + 1 + attr_size,
                hidden_size=128,
                num_layers=2
            )
        elif rnn=='rnn':
            self.rnn = nn.RNN(
                input_size=num_filter + 1 + attr_size,
                hidden_size=128,
                num_layers=1
            )
        if pooling_method=='attention':
            self.attr2atten = nn.Linear(attr_size, 128)
    def out_size(self):
        return 128
    def mean_pooling(self, hiddens, lens):
        hiddens = torch.sum(hiddens, dim = 1, keepdim = False)
        if torch.cuda.is_available():
            lens = torch.cuda.FloatTensor(lens)
        else:
            lens = torch.FloatTensor(lens)
        lens = Variable(torch.unsqueeze(lens, dim = 1), requires_grad = False)
        hiddens = hiddens / lens
        return hiddens
    def attent_pooling(self, hiddens, attr_t):
        hiddens = torch.swapaxes(hiddens, 0, 1)
        attent = torch.tanh(self.attr2atten(attr_t)).unsqueeze(2)
        alpha = torch.bmm(hiddens, attent)
        alpha = torch.exp(-alpha)
        alpha = alpha / torch.sum(alpha, dim=1, keepdim=True)
        hiddens = torch.swapaxes(hiddens, 1, 2)
        hiddens = torch.bmm(hiddens, alpha)
        hiddens = torch.squeeze(hiddens)
        return hiddens
    def forward(self, traj, attr_t):
        conv_locs = self.geo_conv(traj)
        expand_attr_t = attr_t.expand(conv_locs.shape[0],-1,-1)
        conv_locs = torch.cat((conv_locs, expand_attr_t), dim=2)
        # lens = list(map(lambda x: x - self.kernel_size + 1, traj['X_sl']))
        lens = list(traj['X_sl'])
        packed_inputs = nn.utils.rnn.pack_padded_sequence(conv_locs, lens, enforce_sorted=False)
        packed_hiddens, (h_n, c_n) = self.rnn(packed_inputs)
        hiddens, lens = nn.utils.rnn.pad_packed_sequence(packed_hiddens)
        if self.pooling_method == 'mean':
            return packed_hiddens, lens, self.mean_pooling(hiddens, lens)
        elif self.pooling_method == 'attention':
            return packed_hiddens, lens, self.attent_pooling(hiddens, attr_t)