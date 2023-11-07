import torch
from torch.autograd import Variable
import torch.nn as nn

from openbustools.traveltime.models.deeptte import GeoConv


class Net(nn.Module):
    '''
    attr_size: the dimension of attr_net output
    pooling optitions: last, mean, attention
    '''
    def __init__(self, attr_size, kernel_size = 3, num_filter = 32, pooling_method = 'attention', rnn = 'lstm'):
        super(Net, self).__init__()

        self.kernel_size = kernel_size
        self.num_filter = num_filter
        self.pooling_method = pooling_method

        self.geo_conv = GeoConv.Net(kernel_size = kernel_size, num_filter = num_filter)
	    #num_filter: output size of each GeoConv + 1:distance of local path + attr_size: output size of attr component
        if rnn == 'lstm':
                self.rnn = nn.LSTM(input_size = num_filter + 1 + attr_size, \
                                        hidden_size = 128, \
                                        num_layers = 2, \
                                        batch_first = True
                )
        elif rnn == 'rnn':
            self.rnn = nn.RNN(input_size = num_filter + 1 + attr_size, \
                            hidden_size = 128, \
                            num_layers = 1, \
                            batch_first = True
            )


        if pooling_method == 'attention':
            self.attr2atten = nn.Linear(attr_size, 128)

    def out_size(self):
        # return the output size of spatio-temporal component
        return 128

    def mean_pooling(self, hiddens, lens):
        # note that in pad_packed_sequence, the hidden states are padded with all 0
        hiddens = torch.sum(hiddens, dim = 1, keepdim = False)

        if torch.cuda.is_available():
            lens = torch.cuda.FloatTensor(lens)
        else:
            lens = torch.FloatTensor(lens)

        lens = Variable(torch.unsqueeze(lens, dim = 1), requires_grad = False)

        hiddens = hiddens / lens

        return hiddens


    def attent_pooling(self, hiddens, lens, attr_t):
        attent = torch.tanh(self.attr2atten(attr_t)).permute(0, 2, 1)   
        ### Non-linear mapping of learned attribute vector to match hidden_size of LSTM layer
        ### attent size [batch, 128, 1]

	    #hidden b*s*f atten b*f*1 alpha b*s*1 (s is length of sequence)
        alpha = torch.bmm(hiddens, attent)   ### size [batch, max len, 1]
        alpha = torch.exp(-alpha) 

        # The padded hidden is 0 (in pytorch), so we do not need to calculate the mask
        alpha = alpha / torch.sum(alpha, dim = 1, keepdim = True)  
        ### Softmax, although unsure why it takes -alpha above
        ### alpha size [batch, max_len, 1]

        hiddens = hiddens.permute(0, 2, 1)
        hiddens = torch.bmm(hiddens, alpha)
        hiddens = torch.squeeze(hiddens)   ### size [batch, 128]

        return hiddens


    def forward(self, traj, attr_t, config):
        conv_locs = self.geo_conv(traj, config)   ### size [batch, max len of batch traj, kernel_size + 1]

        attr_t = torch.unsqueeze(attr_t, dim = 1)   ### size [batch, 1, 28]
        expand_attr_t = attr_t.expand(conv_locs.size()[:2] + (attr_t.size()[-1], ))   ### size [batch, max len, 28]

        # concat the loc_conv and the attributes
        ### "copy paste" the learned attributes to the Geo-Conv output
        conv_locs = torch.cat((conv_locs, expand_attr_t), dim = 2)   ### size [batch, max len, kernel_size + 1 + 28]
        
        lens = list(map(lambda x: x - self.kernel_size + 1, traj['lens']))   ### size [batch, ]

        packed_inputs = nn.utils.rnn.pack_padded_sequence(conv_locs, lens, batch_first = True, enforce_sorted=False)
        ### pack_padded_sequence: https://stackoverflow.com/questions/51030782/why-do-we-pack-the-sequences-in-pytorch
        ### Basically it is used for computational optimisation of padded sequences

        packed_hiddens, (h_n, c_n) = self.rnn(packed_inputs)
        ### packed_hiddens is packed hidden states of all time steps
        ### h_n and c_n are packed hidden and cell states from both LSTM layers resp. at last time step

        hiddens, lens = nn.utils.rnn.pad_packed_sequence(packed_hiddens, batch_first = True)
        ### hiddens size [batch, max batch traj len, 128]
        ### lens size [batch, ] (array of lengths of unpadded sequence)

        if self.pooling_method == 'mean':
            return packed_hiddens, lens, self.mean_pooling(hiddens, lens)

        elif self.pooling_method == 'attention':
            ### self.attent_pooling() size [batch, 128]
            return packed_hiddens, lens, self.attent_pooling(hiddens, lens, attr_t)
