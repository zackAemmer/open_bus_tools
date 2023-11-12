import torch
import torch.nn as nn

from openbustools.traveltime.models import embedding


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.embed_dims_cfg = embedding.EMBED_DICT
        self.build()
    def build(self):
        for k,val in self.embed_dims_cfg.items():
            self.add_module(k + '_em', nn.Embedding(val['vocab_dim'], val['embed_dim']))
    def out_size(self):
        sz = 0
        for _,val in self.embed_dims_cfg.items():
            sz += val['embed_dim']
        return sz + 1
    def forward(self, attr):
        em_list = []
        for name,_ in self.embed_dims_cfg.items():
            embed = getattr(self, name + '_em')
            attr_t = attr[name].view(-1, 1)
            attr_t = torch.squeeze(embed(attr_t))
            # The model breaks here if batch size is 1
            if len(attr_t.shape)==1:
                attr_t = attr_t.view(1,-1)
            em_list.append(attr_t)
        em_list.append(attr['cumul_dist_km'].view(-1, 1))
        return torch.cat(em_list, dim=1)