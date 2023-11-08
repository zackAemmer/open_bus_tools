import torch
import torch.nn as nn
import torch.nn.functional as F

from openbustools.traveltime import model_utils
import numpy as np

from torch.autograd import Variable

class Net(nn.Module):
    ### (key, total possible values of key, embedded dimension)
    ### dateID isnt embedded?
    # embed_dims = [('driverID', 1152, 16), ('weekID', 7, 3), ('timeID', 1440, 8)]

    def __init__(self):
        super(Net, self).__init__()
        # self.embed_dims_cfg = [('vehicleID', cfg['n_unique_veh'], 16), ('weekID', 7, 3), ('timeID', 1440, 8)]
        self.embed_dims_cfg = [('weekID', 7, 3), ('timeID', 1440, 8)]
        # whether to add the two ends of the path into Attribute Component
        self.build()

    def build(self):
        for name, dim_in, dim_out in self.embed_dims_cfg:
            self.add_module(name + '_em', nn.Embedding(dim_in, dim_out))
            ### Embedding: https://discuss.pytorch.org/t/what-is-nn-embedding-exactly-doing/12521/9 
            ### 3 different embedding layers, 1 for each feature

    def out_size(self):
        ### Used for building ST component beforehand
        sz = 0
        for name, dim_in, dim_out in self.embed_dims_cfg:
            sz += dim_out
        # append total distance
        return sz + 1

    def forward(self, attr):
        em_list = []
        for name, dim_in, dim_out in self.embed_dims_cfg:
            embed = getattr(self, name + '_em')
            attr_t = attr[name].view(-1, 1)   ### Original is 1D, reshape to 2D
            attr_t = torch.squeeze(embed(attr_t))   ### embed(attr_t) is size [batch, 1, dim_out]
            # The model breaks here if batch size is 1
            if len(attr_t.shape)==1:
                attr_t = attr_t.view(1,-1)
            em_list.append(attr_t)

        # dist = utils.normalize(attr['dist'], 'dist')   ### Not sure why this is normalised again, alr done in dataloader
        em_list.append(attr['dist'].view(-1, 1))

        return torch.cat(em_list, dim = 1)
        ### output dim [batch, 28] ; 28 = 16 driverID, 3 weekID, 8 timeID, 1 dist