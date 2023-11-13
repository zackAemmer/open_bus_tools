import numpy as np
import torch
from torch import nn
import lightning.pytorch as pl

from openbustools import data_utils
from openbustools.traveltime import masked_loss, model_utils
from openbustools.traveltime.models import embedding


class CONV(pl.LightningModule):
    def __init__(self, model_name, config, holdout_routes, input_size, collate_fn, batch_size, hidden_size, num_layers, dropout_rate):
        super(CONV, self).__init__()
        self.save_hyperparameters()
        self.model_name = model_name
        self.config = config
        self.holdout_routes = holdout_routes
        self.input_size = input_size
        self.collate_fn = collate_fn
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.is_nn = True
        self.loss_fn = masked_loss.MaskedHuberLoss()
        # Embeddings
        self.min_em = embedding.MinuteEmbedding()
        self.day_em = embedding.DayEmbedding()
        self.embed_total_dims = self.min_em.embed_dim + self.day_em.embed_dim
        # 1D-Convolution
        self.conv1d = nn.Sequential()
        self.conv1d.append(nn.Conv1d(in_channels=self.input_size, out_channels=self.hidden_size, kernel_size=3, padding=1))
        self.conv1d.append(nn.ReLU())
        for _ in range(self.num_layers):
            self.conv1d.append(nn.Conv1d(in_channels=self.hidden_size, out_channels=self.hidden_size, kernel_size=3, padding=1))
            self.conv1d.append(nn.ReLU())
        self.conv1d.append(nn.Dropout(p=self.dropout_rate))
        # Linear compression/feature extraction
        self.feature_extract = nn.Linear(in_features=self.hidden_size + self.embed_total_dims, out_features=1)
        self.feature_extract_activation = nn.ReLU()
    def training_step(self, batch):
        x,y = batch
        x_em = x[0]
        x_ct = x[1]
        x_sl = x[2]
        # Embed categorical variables
        x_min_em = self.min_em(x_em[:,0])
        x_day_em = self.day_em(x_em[:,1])
        x_em = torch.cat((x_min_em, x_day_em), dim=1).unsqueeze(0)
        x_em = x_em.expand(x_ct.shape[0],-1,-1)
        # Get conv pred
        x_ct = torch.swapaxes(x_ct, 0, 1)
        x_ct = torch.swapaxes(x_ct, 1, 2)
        x_ct = self.conv1d(x_ct)
        x_ct = torch.swapaxes(x_ct, 0, 1)
        x_ct = torch.swapaxes(x_ct, 0, 2)
        # Combine all variables
        out = torch.cat([x_em, x_ct], dim=2)
        out = self.feature_extract(self.feature_extract_activation(out)).squeeze(2)
        # Masked loss; init mask here since module knows device
        mask = torch.zeros(max(x_sl), len(x_sl), dtype=torch.bool, device=self.device)
        mask = model_utils.fill_tensor_mask(mask, x_sl)
        loss = self.loss_fn(out, y, mask)
        self.log_dict(
            {'train_loss': loss},
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=self.batch_size
        )
        return loss
    def validation_step(self, batch):
        x,y = batch
        x_em = x[0]
        x_ct = x[1]
        x_sl = x[2]
        # Embed categorical variables
        x_min_em = self.min_em(x_em[:,0])
        x_day_em = self.day_em(x_em[:,1])
        x_em = torch.cat((x_min_em, x_day_em), dim=1).unsqueeze(0)
        x_em = x_em.expand(x_ct.shape[0],-1,-1)
        # Get conv pred
        x_ct = torch.swapaxes(x_ct, 0, 1)
        x_ct = torch.swapaxes(x_ct, 1, 2)
        x_ct = self.conv1d(x_ct)
        x_ct = torch.swapaxes(x_ct, 0, 1)
        x_ct = torch.swapaxes(x_ct, 0, 2)
        # Combine all variables
        out = torch.cat([x_em, x_ct], dim=2)
        out = self.feature_extract(self.feature_extract_activation(out)).squeeze(2)
        # Masked loss; init mask here since module knows device
        mask = torch.zeros(max(x_sl), len(x_sl), dtype=torch.bool, device=self.device)
        mask = model_utils.fill_tensor_mask(mask, x_sl)
        loss = self.loss_fn(out, y, mask)
        self.log_dict(
            {'valid_loss': loss},
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=self.batch_size
        )
        return loss
    def predict_step(self, batch):
        x,y = batch
        x_em = x[0]
        x_ct = x[1]
        x_sl = x[2]
        # Embed categorical variables
        x_min_em = self.min_em(x_em[:,0])
        x_day_em = self.day_em(x_em[:,1])
        x_em = torch.cat((x_min_em, x_day_em), dim=1).unsqueeze(0)
        x_em = x_em.expand(x_ct.shape[0], -1, -1)
        # Get conv pred
        x_ct = torch.swapaxes(x_ct, 0, 1)
        x_ct = torch.swapaxes(x_ct, 1, 2)
        x_ct = self.conv1d(x_ct)
        x_ct = torch.swapaxes(x_ct, 0, 1)
        x_ct = torch.swapaxes(x_ct, 0, 2)
        # Combine all variables
        out = torch.cat([x_em, x_ct], dim=2)
        out = self.feature_extract(self.feature_extract_activation(out)).squeeze(2)
        # Masked loss; init mask here since module knows device
        mask = torch.zeros(max(x_sl), len(x_sl), dtype=torch.bool, device=self.device)
        mask = model_utils.fill_tensor_mask(mask, x_sl)
        # Move to cpu and return predictions, labels
        mask = mask.detach().cpu().numpy()
        out = out.detach().cpu().numpy()
        y = y.detach().cpu().numpy()
        out_agg = model_utils.aggregate_tts(out, mask)
        y_agg = model_utils.aggregate_tts(y, mask)
        return {'preds': out_agg, 'labels': y_agg}
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


# class CONV_REALTIME(pl.LightningModule):
#     def __init__(self, model_name, n_features, n_grid_features, grid_compression_size, hyperparameter_dict, embed_dict, collate_fn, config):
#         super(CONV_REALTIME, self).__init__()
#         self.save_hyperparameters()
#         self.model_name = model_name
#         self.n_features = n_features
#         self.n_grid_features = n_grid_features
#         self.grid_compression_size = grid_compression_size
#         self.hyperparameter_dict = hyperparameter_dict
#         self.batch_size = int(self.hyperparameter_dict['batch_size'])
#         self.embed_dict = embed_dict
#         self.collate_fn = collate_fn
#         self.config = config
#         self.is_nn = True
#         self.requires_grid = True
#         self.train_time = 0.0
#         self.loss_fn = masked_loss.MaskedHuberLoss()
#         # Embeddings
#         self.embed_total_dims = np.sum([self.embed_dict[key]['embed_dims'] for key in self.embed_dict.keys()]).astype('int32')
#         self.timeID_em = nn.Embedding(self.embed_dict['timeID']['vocab_size'], self.embed_dict['timeID']['embed_dims'])
#         self.weekID_em = nn.Embedding(self.embed_dict['weekID']['vocab_size'], self.embed_dict['weekID']['embed_dims'])
#         # Grid Feedforward
#         self.grid_norm = nn.BatchNorm1d(self.n_grid_features)
#         self.linear_relu_stack_grid = nn.Sequential(
#             nn.Linear(self.n_grid_features, self.hyperparameter_dict['hidden_size']),
#             nn.ReLU(),
#             nn.Linear(self.hyperparameter_dict['hidden_size'], self.grid_compression_size),
#             nn.ReLU()
#         )
#         # Conv1d layer
#         self.conv1d = nn.Sequential()
#         self.conv1d.append(nn.BatchNorm1d(self.n_features + self.grid_compression_size))
#         self.conv1d.append(nn.Conv1d(in_channels=self.n_features + self.grid_compression_size, out_channels=self.hyperparameter_dict['hidden_size'], kernel_size=3, padding=1))
#         self.conv1d.append(nn.ReLU())
#         for i in range(self.hyperparameter_dict['num_layers']):
#             self.conv1d.append(nn.Conv1d(in_channels=self.hyperparameter_dict['hidden_size'], out_channels=self.hyperparameter_dict['hidden_size'], kernel_size=3, padding=1))
#             self.conv1d.append(nn.ReLU())
#         self.conv1d.append(nn.Dropout(p=self.hyperparameter_dict['dropout_rate']))
#         # Linear compression layer
#         self.feature_extract = nn.Linear(in_features=self.hyperparameter_dict['hidden_size'] + self.embed_total_dims, out_features=1)
#         self.feature_extract_activation = nn.ReLU()
#     def training_step(self, batch, batch_idx):
#         x,y = batch
#         x_em = x[0]
#         x_ct = x[1]
#         x_gr = x[2]
#         x_sl = x[3]
#         # Embed categorical variables
#         timeID_embedded = self.timeID_em(x_em[:,0])
#         weekID_embedded = self.weekID_em(x_em[:,1])
#         x_em = torch.cat((timeID_embedded,weekID_embedded), dim=1).unsqueeze(1)
#         x_em = x_em.expand(-1, x_ct.shape[1], -1)
#         x_em = torch.swapaxes(x_em, 1, 2)
#         # Feed grid data through model
#         x_gr = torch.flatten(x_gr, 2)
#         x_gr = torch.swapaxes(x_gr, 1, 2)
#         x_gr = self.grid_norm(x_gr)
#         x_gr = torch.swapaxes(x_gr, 1, 2)
#         x_gr = self.linear_relu_stack_grid(x_gr)
#         # Get conv pred
#         x_ct = torch.cat([x_ct, x_gr], dim=2)
#         x_ct = torch.swapaxes(x_ct, 1, 2)
#         x_ct = self.conv1d(x_ct)
#         # Combine all variables
#         out = torch.cat([x_em, x_ct], dim=1)
#         out = torch.swapaxes(out, 1, 2)
#         out = self.feature_extract(self.feature_extract_activation(out)).squeeze(2)
#         mask = data_utils.create_tensor_mask(x_sl, self.device)
#         loss = self.loss_fn(out, y, mask)
#         self.log_dict(
#             {
#                 'train_loss': loss,
#             },
#             on_step=False,
#             on_epoch=True,
#             prog_bar=True,
#         )
#         # for name, param in self.named_parameters():
#         #     self.logger.experiment.add_histogram(name, param, self.current_epoch)
#         return loss
#     def validation_step(self, batch, batch_idx):
#         x,y = batch
#         x_em = x[0]
#         x_ct = x[1]
#         x_gr = x[2]
#         x_sl = x[3]
#         # Embed categorical variables
#         timeID_embedded = self.timeID_em(x_em[:,0])
#         weekID_embedded = self.weekID_em(x_em[:,1])
#         x_em = torch.cat((timeID_embedded,weekID_embedded), dim=1).unsqueeze(1)
#         x_em = x_em.expand(-1, x_ct.shape[1], -1)
#         x_em = torch.swapaxes(x_em, 1, 2)
#         # Feed grid data through model
#         x_gr = torch.flatten(x_gr, 2)
#         x_gr = torch.swapaxes(x_gr, 1, 2)
#         x_gr = self.grid_norm(x_gr)
#         x_gr = torch.swapaxes(x_gr, 1, 2)
#         x_gr = self.linear_relu_stack_grid(x_gr)
#         # Get conv pred
#         x_ct = torch.cat([x_ct, x_gr], dim=2)
#         x_ct = torch.swapaxes(x_ct, 1, 2)
#         x_ct = self.conv1d(x_ct)
#         # Combine all variables
#         out = torch.cat([x_em, x_ct], dim=1)
#         out = torch.swapaxes(out, 1, 2)
#         out = self.feature_extract(self.feature_extract_activation(out)).squeeze(2)
#         mask = data_utils.create_tensor_mask(x_sl, self.device)
#         loss = self.loss_fn(out, y, mask)
#         self.log_dict(
#             {
#                 'valid_loss': loss,
#             },
#             on_step=False,
#             on_epoch=True,
#             prog_bar=True,
#         )
#         return loss
#     def predict_step(self, batch, batch_idx):
#         x,y = batch
#         x_em = x[0]
#         x_ct = x[1]
#         x_gr = x[2]
#         x_sl = x[3]
#         # Embed categorical variables
#         timeID_embedded = self.timeID_em(x_em[:,0])
#         weekID_embedded = self.weekID_em(x_em[:,1])
#         x_em = torch.cat((timeID_embedded,weekID_embedded), dim=1).unsqueeze(1)
#         x_em = x_em.expand(-1, x_ct.shape[1], -1)
#         x_em = torch.swapaxes(x_em, 1, 2)
#         # Feed grid data through model
#         x_gr = torch.flatten(x_gr, 2)
#         x_gr = torch.swapaxes(x_gr, 1, 2)
#         x_gr = self.grid_norm(x_gr)
#         x_gr = torch.swapaxes(x_gr, 1, 2)
#         x_gr = self.linear_relu_stack_grid(x_gr)
#         # Get conv pred
#         x_ct = torch.cat([x_ct, x_gr], dim=2)
#         x_ct = torch.swapaxes(x_ct, 1, 2)
#         x_ct = self.conv1d(x_ct)
#         # Combine all variables
#         out = torch.cat([x_em, x_ct], dim=1)
#         out = torch.swapaxes(out, 1, 2)
#         out = self.feature_extract(self.feature_extract_activation(out)).squeeze(2)
#         mask = data_utils.create_tensor_mask(x_sl, self.device, drop_first=False)
#         mask = mask.detach().cpu().numpy()
#         out  = (out.detach().cpu().numpy() * self.config['time_calc_s_std']) + self.config['time_calc_s_mean']
#         y = (y.detach().cpu().numpy() * self.config['time_calc_s_std']) + self.config['time_calc_s_mean']
#         out_agg = data_utils.aggregate_tts(out, mask)
#         y_agg = data_utils.aggregate_tts(y, mask)
#         return {"out_agg":out_agg, "y_agg":y_agg, "out":out, "y":y, "mask":mask}
#     def configure_optimizers(self):
#         optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
#         return optimizer