import numpy as np
import torch
from torch import nn
import lightning.pytorch as pl

from openbustools import data_utils
from openbustools.traveltime import masked_loss


class CONV_L(pl.LightningModule):
    def __init__(self, model_name, n_features, hyperparameter_dict, embed_dict, collate_fn, config):
        super(CONV_L, self).__init__()
        self.save_hyperparameters()
        self.model_name = model_name
        self.n_features = n_features
        self.hyperparameter_dict = hyperparameter_dict
        self.batch_size = int(self.hyperparameter_dict['batch_size'])
        self.embed_dict = embed_dict
        self.collate_fn = collate_fn
        self.config = config
        self.is_nn = True
        self.requires_grid = False
        self.train_time = 0.0
        self.loss_fn = masked_loss.MaskedHuberLoss()
        # Embeddings
        self.embed_total_dims = np.sum([self.embed_dict[key]['embed_dims'] for key in self.embed_dict.keys()]).astype('int32')
        self.timeID_em = nn.Embedding(self.embed_dict['timeID']['vocab_size'], self.embed_dict['timeID']['embed_dims'])
        self.weekID_em = nn.Embedding(self.embed_dict['weekID']['vocab_size'], self.embed_dict['weekID']['embed_dims'])
        # Conv1d layer
        self.conv1d = nn.Sequential()
        self.conv1d.append(nn.BatchNorm1d(self.n_features))
        self.conv1d.append(nn.Conv1d(in_channels=self.n_features, out_channels=self.hyperparameter_dict['hidden_size'], kernel_size=3, padding=1))
        self.conv1d.append(nn.ReLU())
        for i in range(self.hyperparameter_dict['num_layers']):
            self.conv1d.append(nn.Conv1d(in_channels=self.hyperparameter_dict['hidden_size'], out_channels=self.hyperparameter_dict['hidden_size'], kernel_size=3, padding=1))
            self.conv1d.append(nn.ReLU())
        self.conv1d.append(nn.Dropout(p=self.hyperparameter_dict['dropout_rate']))
        # Linear compression layer
        self.feature_extract = nn.Linear(in_features=self.hyperparameter_dict['hidden_size'] + self.embed_total_dims, out_features=1)
        self.feature_extract_activation = nn.ReLU()
    def training_step(self, batch, batch_idx):
        x,y = batch
        x_em = x[0]
        x_ct = x[1]
        x_sl = x[2]
        # Embed categorical variables
        timeID_embedded = self.timeID_em(x_em[:,0])
        weekID_embedded = self.weekID_em(x_em[:,1])
        x_em = torch.cat((timeID_embedded,weekID_embedded), dim=1).unsqueeze(1)
        x_em = x_em.expand(-1, x_ct.shape[1], -1)
        x_em = torch.swapaxes(x_em, 1, 2)
        # Get conv pred
        x_ct = torch.swapaxes(x_ct, 1, 2)
        x_ct = self.conv1d(x_ct)
        # Combine all variables
        out = torch.cat([x_em, x_ct], dim=1)
        out = torch.swapaxes(out, 1, 2)
        out = self.feature_extract(self.feature_extract_activation(out)).squeeze(2)
        mask = data_utils.create_tensor_mask(x_sl, self.device)
        loss = self.loss_fn(out, y, mask)
        self.log_dict(
            {
                'train_loss': loss,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        # for name, param in self.named_parameters():
        #     self.logger.experiment.add_histogram(name, param, self.current_epoch)
        return loss
    def validation_step(self, batch, batch_idx):
        x,y = batch
        x_em = x[0]
        x_ct = x[1]
        x_sl = x[2]
        # Embed categorical variables
        timeID_embedded = self.timeID_em(x_em[:,0])
        weekID_embedded = self.weekID_em(x_em[:,1])
        x_em = torch.cat((timeID_embedded,weekID_embedded), dim=1).unsqueeze(1)
        x_em = x_em.expand(-1, x_ct.shape[1], -1)
        x_em = torch.swapaxes(x_em, 1, 2)
        # Get conv pred
        x_ct = torch.swapaxes(x_ct, 1, 2)
        x_ct = self.conv1d(x_ct)
        # Combine all variables
        out = torch.cat([x_em, x_ct], dim=1)
        out = torch.swapaxes(out, 1, 2)
        out = self.feature_extract(self.feature_extract_activation(out)).squeeze(2)
        mask = data_utils.create_tensor_mask(x_sl, self.device)
        loss = self.loss_fn(out, y, mask)
        self.log_dict(
            {
                'valid_loss': loss,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return loss
    def predict_step(self, batch, batch_idx):
        x,y = batch
        x_em = x[0]
        x_ct = x[1]
        x_sl = x[2]
        # Embed categorical variables
        timeID_embedded = self.timeID_em(x_em[:,0])
        weekID_embedded = self.weekID_em(x_em[:,1])
        x_em = torch.cat((timeID_embedded,weekID_embedded), dim=1).unsqueeze(1)
        x_em = x_em.expand(-1, x_ct.shape[1], -1)
        x_em = torch.swapaxes(x_em, 1, 2)
        # Get conv pred
        x_ct = torch.swapaxes(x_ct, 1, 2)
        x_ct = self.conv1d(x_ct)
        # Combine all variables
        out = torch.cat([x_em, x_ct], dim=1)
        out = torch.swapaxes(out, 1, 2)
        out = self.feature_extract(self.feature_extract_activation(out)).squeeze(2)
        mask = data_utils.create_tensor_mask(x_sl, self.device, drop_first=False)
        mask = mask.detach().cpu().numpy()
        out  = (out.detach().cpu().numpy() * self.config['time_calc_s_std']) + self.config['time_calc_s_mean']
        y = (y.detach().cpu().numpy() * self.config['time_calc_s_std']) + self.config['time_calc_s_mean']
        out_agg = data_utils.aggregate_tts(out, mask)
        y_agg = data_utils.aggregate_tts(y, mask)
        return {"out_agg":out_agg, "y_agg":y_agg, "out":out, "y":y, "mask":mask}
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

class CONV_GRID_L(pl.LightningModule):
    def __init__(self, model_name, n_features, n_grid_features, grid_compression_size, hyperparameter_dict, embed_dict, collate_fn, config):
        super(CONV_GRID_L, self).__init__()
        self.save_hyperparameters()
        self.model_name = model_name
        self.n_features = n_features
        self.n_grid_features = n_grid_features
        self.grid_compression_size = grid_compression_size
        self.hyperparameter_dict = hyperparameter_dict
        self.batch_size = int(self.hyperparameter_dict['batch_size'])
        self.embed_dict = embed_dict
        self.collate_fn = collate_fn
        self.config = config
        self.is_nn = True
        self.requires_grid = True
        self.train_time = 0.0
        self.loss_fn = masked_loss.MaskedHuberLoss()
        # Embeddings
        self.embed_total_dims = np.sum([self.embed_dict[key]['embed_dims'] for key in self.embed_dict.keys()]).astype('int32')
        self.timeID_em = nn.Embedding(self.embed_dict['timeID']['vocab_size'], self.embed_dict['timeID']['embed_dims'])
        self.weekID_em = nn.Embedding(self.embed_dict['weekID']['vocab_size'], self.embed_dict['weekID']['embed_dims'])
        # Grid Feedforward
        self.grid_norm = nn.BatchNorm1d(self.n_grid_features)
        self.linear_relu_stack_grid = nn.Sequential(
            nn.Linear(self.n_grid_features, self.hyperparameter_dict['hidden_size']),
            nn.ReLU(),
            nn.Linear(self.hyperparameter_dict['hidden_size'], self.grid_compression_size),
            nn.ReLU()
        )
        # Conv1d layer
        self.conv1d = nn.Sequential()
        self.conv1d.append(nn.BatchNorm1d(self.n_features + self.grid_compression_size))
        self.conv1d.append(nn.Conv1d(in_channels=self.n_features + self.grid_compression_size, out_channels=self.hyperparameter_dict['hidden_size'], kernel_size=3, padding=1))
        self.conv1d.append(nn.ReLU())
        for i in range(self.hyperparameter_dict['num_layers']):
            self.conv1d.append(nn.Conv1d(in_channels=self.hyperparameter_dict['hidden_size'], out_channels=self.hyperparameter_dict['hidden_size'], kernel_size=3, padding=1))
            self.conv1d.append(nn.ReLU())
        self.conv1d.append(nn.Dropout(p=self.hyperparameter_dict['dropout_rate']))
        # Linear compression layer
        self.feature_extract = nn.Linear(in_features=self.hyperparameter_dict['hidden_size'] + self.embed_total_dims, out_features=1)
        self.feature_extract_activation = nn.ReLU()
    def training_step(self, batch, batch_idx):
        x,y = batch
        x_em = x[0]
        x_ct = x[1]
        x_gr = x[2]
        x_sl = x[3]
        # Embed categorical variables
        timeID_embedded = self.timeID_em(x_em[:,0])
        weekID_embedded = self.weekID_em(x_em[:,1])
        x_em = torch.cat((timeID_embedded,weekID_embedded), dim=1).unsqueeze(1)
        x_em = x_em.expand(-1, x_ct.shape[1], -1)
        x_em = torch.swapaxes(x_em, 1, 2)
        # Feed grid data through model
        x_gr = torch.flatten(x_gr, 2)
        x_gr = torch.swapaxes(x_gr, 1, 2)
        x_gr = self.grid_norm(x_gr)
        x_gr = torch.swapaxes(x_gr, 1, 2)
        x_gr = self.linear_relu_stack_grid(x_gr)
        # Get conv pred
        x_ct = torch.cat([x_ct, x_gr], dim=2)
        x_ct = torch.swapaxes(x_ct, 1, 2)
        x_ct = self.conv1d(x_ct)
        # Combine all variables
        out = torch.cat([x_em, x_ct], dim=1)
        out = torch.swapaxes(out, 1, 2)
        out = self.feature_extract(self.feature_extract_activation(out)).squeeze(2)
        mask = data_utils.create_tensor_mask(x_sl, self.device)
        loss = self.loss_fn(out, y, mask)
        self.log_dict(
            {
                'train_loss': loss,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        # for name, param in self.named_parameters():
        #     self.logger.experiment.add_histogram(name, param, self.current_epoch)
        return loss
    def validation_step(self, batch, batch_idx):
        x,y = batch
        x_em = x[0]
        x_ct = x[1]
        x_gr = x[2]
        x_sl = x[3]
        # Embed categorical variables
        timeID_embedded = self.timeID_em(x_em[:,0])
        weekID_embedded = self.weekID_em(x_em[:,1])
        x_em = torch.cat((timeID_embedded,weekID_embedded), dim=1).unsqueeze(1)
        x_em = x_em.expand(-1, x_ct.shape[1], -1)
        x_em = torch.swapaxes(x_em, 1, 2)
        # Feed grid data through model
        x_gr = torch.flatten(x_gr, 2)
        x_gr = torch.swapaxes(x_gr, 1, 2)
        x_gr = self.grid_norm(x_gr)
        x_gr = torch.swapaxes(x_gr, 1, 2)
        x_gr = self.linear_relu_stack_grid(x_gr)
        # Get conv pred
        x_ct = torch.cat([x_ct, x_gr], dim=2)
        x_ct = torch.swapaxes(x_ct, 1, 2)
        x_ct = self.conv1d(x_ct)
        # Combine all variables
        out = torch.cat([x_em, x_ct], dim=1)
        out = torch.swapaxes(out, 1, 2)
        out = self.feature_extract(self.feature_extract_activation(out)).squeeze(2)
        mask = data_utils.create_tensor_mask(x_sl, self.device)
        loss = self.loss_fn(out, y, mask)
        self.log_dict(
            {
                'valid_loss': loss,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return loss
    def predict_step(self, batch, batch_idx):
        x,y = batch
        x_em = x[0]
        x_ct = x[1]
        x_gr = x[2]
        x_sl = x[3]
        # Embed categorical variables
        timeID_embedded = self.timeID_em(x_em[:,0])
        weekID_embedded = self.weekID_em(x_em[:,1])
        x_em = torch.cat((timeID_embedded,weekID_embedded), dim=1).unsqueeze(1)
        x_em = x_em.expand(-1, x_ct.shape[1], -1)
        x_em = torch.swapaxes(x_em, 1, 2)
        # Feed grid data through model
        x_gr = torch.flatten(x_gr, 2)
        x_gr = torch.swapaxes(x_gr, 1, 2)
        x_gr = self.grid_norm(x_gr)
        x_gr = torch.swapaxes(x_gr, 1, 2)
        x_gr = self.linear_relu_stack_grid(x_gr)
        # Get conv pred
        x_ct = torch.cat([x_ct, x_gr], dim=2)
        x_ct = torch.swapaxes(x_ct, 1, 2)
        x_ct = self.conv1d(x_ct)
        # Combine all variables
        out = torch.cat([x_em, x_ct], dim=1)
        out = torch.swapaxes(out, 1, 2)
        out = self.feature_extract(self.feature_extract_activation(out)).squeeze(2)
        mask = data_utils.create_tensor_mask(x_sl, self.device, drop_first=False)
        mask = mask.detach().cpu().numpy()
        out  = (out.detach().cpu().numpy() * self.config['time_calc_s_std']) + self.config['time_calc_s_mean']
        y = (y.detach().cpu().numpy() * self.config['time_calc_s_std']) + self.config['time_calc_s_mean']
        out_agg = data_utils.aggregate_tts(out, mask)
        y_agg = data_utils.aggregate_tts(y, mask)
        return {"out_agg":out_agg, "y_agg":y_agg, "out":out, "y":y, "mask":mask}
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer