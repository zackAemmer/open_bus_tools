import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl

from openbustools.traveltime import model_utils
from openbustools.traveltime.models.deeptte import Attr, SpatioTemporal, deeptte_utils


class EntireEstimator(nn.Module):
    def __init__(self, input_size, num_final_fcs, hidden_size = 128):
        super(EntireEstimator, self).__init__()
        self.input2hid = nn.Linear(input_size, hidden_size)
        self.residuals = nn.ModuleList()
        for i in range(num_final_fcs):
            self.residuals.append(nn.Linear(hidden_size, hidden_size))
        # Previously named hid2out; renamed for fine-tuning purposes
        self.feature_extract = nn.Linear(hidden_size, 1)
    def forward(self, attr_t, sptm_t):
        # Breaks on batch with 1
        if len(sptm_t.shape)==1:
            sptm_t = sptm_t.view(1,-1)
        inputs = torch.cat((attr_t, sptm_t), dim = 1)
        hidden = F.leaky_relu(self.input2hid(inputs))
        for i in range(len(self.residuals)):
            residual = F.leaky_relu(self.residuals[i](hidden))
            hidden = hidden + residual
        out = self.feature_extract(hidden)
        return out
    def eval_on_batch(self, pred, label):
        loss_fn = nn.HuberLoss()
        loss = loss_fn(label, pred)
        return {'label': label.detach().cpu().numpy(), 'pred': pred.detach().cpu().numpy()}, loss


class LocalEstimator(nn.Module):
    def __init__(self, input_size):
        super(LocalEstimator, self).__init__()
        self.input2hid = nn.Linear(input_size, 64)
        self.hid2hid = nn.Linear(64, 32)
        # Previously hid2out, renamed for fine tuning
        self.feature_extract = nn.Linear(32, 1)
    def forward(self, sptm_s):
        hidden = F.leaky_relu(self.input2hid(sptm_s))
        hidden = F.leaky_relu(self.hid2hid(hidden))
        out = self.feature_extract(hidden)
        return out
    def eval_on_batch(self, pred, label):
        loss_fn = nn.HuberLoss()
        loss = loss_fn(label, pred)
        return loss


class Net(pl.LightningModule):
    def __init__(self, model_name, config, holdout_routes, input_size, collate_fn, batch_size, kernel_size=3, num_filter=32, pooling_method='attention', num_final_fcs=3, final_fc_size=128, alpha=0.3):
        super(Net, self).__init__()
        self.save_hyperparameters()
        # Training configurations
        self.model_name = model_name
        self.config = config
        self.holdout_routes = holdout_routes
        self.input_size = input_size
        self.collate_fn = collate_fn
        self.batch_size = batch_size
        self.is_nn = True
        # Parameter of attribute / spatio-temporal component
        self.kernel_size = kernel_size
        self.num_filter = num_filter
        self.pooling_method = pooling_method
        # Parameter of multi-task learning component
        self.num_final_fcs = num_final_fcs
        self.final_fc_size = final_fc_size
        self.alpha = alpha
        self.build()
        self.init_weight()
    def build(self):
        self.attr_net = Attr.Net()
        self.spatio_temporal = SpatioTemporal.Net(attr_size=self.attr_net.out_size(), kernel_size=self.kernel_size, num_filter=self.num_filter, pooling_method=self.pooling_method)
        self.entire_estimate = EntireEstimator(input_size=self.spatio_temporal.out_size() + self.attr_net.out_size(), num_final_fcs=self.num_final_fcs, hidden_size=self.final_fc_size)
        self.local_estimate = LocalEstimator(input_size=self.spatio_temporal.out_size())
    def init_weight(self):
        for name, param in self.named_parameters():
            if name.find('.bias') != -1:
                param.data.fill_(0)
            elif name.find('.weight') != -1:
                nn.init.xavier_uniform_(param.data)
    def forward(self, attr, traj):
        attr_t = self.attr_net(attr)
        sptm_s, sptm_l, sptm_t = self.spatio_temporal(traj, attr_t)
        entire_out = self.entire_estimate(attr_t, sptm_t)
        if self.training:
            local_out = self.local_estimate(sptm_s[0])
            return entire_out, (local_out, sptm_l)
        else:
            return entire_out
    def training_step(self, batch):
        attr = batch[0]
        traj = batch[1]
        # Estimates for full trajectory and individual points
        entire_out, (local_out, local_length) = self(attr, traj)
        _, entire_loss = self.entire_estimate.eval_on_batch(entire_out.squeeze(), attr['cumul_time_s'])
        # Un-masked 1d array of labels for individual points
        mask = torch.zeros(max(local_length), len(local_length), dtype=torch.bool, device=self.device)
        mask = model_utils.fill_tensor_mask(mask, local_length, drop_first=False)
        local_label = traj['calc_time_s']
        local_label = torch.masked_select(local_label, mask)
        local_loss = self.local_estimate.eval_on_batch(local_out.squeeze(), local_label)
        # According to eqn 8 of paper
        loss = (1 - self.alpha) * entire_loss + self.alpha * local_loss
        self.log_dict(
            {'train_loss': loss},
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return loss
    def validation_step(self, batch):
        attr = batch[0]
        traj = batch[1]
        entire_out = self(attr, traj).squeeze()
        pred_dict, entire_loss = self.entire_estimate.eval_on_batch(entire_out, attr['cumul_time_s'])
        self.log_dict(
            {'valid_loss': entire_loss},
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=self.batch_size
        )
        return entire_loss
    def predict_step(self, batch):
        attr = batch[0]
        traj = batch[1]
        entire_out = self(attr, traj).squeeze()
        pred_dict, entire_loss = self.entire_estimate.eval_on_batch(entire_out, attr['cumul_time_s'])
        return {'preds': pred_dict['pred'], 'labels': pred_dict['label']}
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer