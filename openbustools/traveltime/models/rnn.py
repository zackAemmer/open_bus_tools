import numpy as np
import torch
from torch import nn
import lightning.pytorch as pl

from openbustools.traveltime import masked_loss, model_utils, data_loader
from openbustools.traveltime.models import embedding, realtime


class GRU(pl.LightningModule):
    def __init__(self, model_name, config, holdout_routes, input_size, collate_fn, batch_size, hidden_size, num_layers, dropout_rate):
        super(GRU, self).__init__()
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
        self.include_grid = False
        self.loss_fn = torch.nn.MSELoss()
        # Embeddings
        self.min_em = embedding.MinuteEmbedding()
        self.day_em = embedding.DayEmbedding()
        self.embed_total_dims = self.min_em.embed_dim + self.day_em.embed_dim
        # Recurrent
        self.rnn = nn.GRU(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers, dropout=self.dropout_rate)
        # Linear compression/feature extraction
        self.feature_extract = nn.Linear(in_features=self.hidden_size + self.embed_total_dims, out_features=1)
        self.feature_extract_activation = nn.ReLU()
    def forward(self, x):
        x_em, x_ct = x
        # Embed categorical variables
        x_min_em = self.min_em(x_em[:,:,0])
        x_day_em = self.day_em(x_em[:,:,1])
        # Get recurrent pred
        x_ct, hidden_prev = self.rnn(x_ct)
        # Combine all variables
        out = torch.cat([x_ct, x_min_em, x_day_em], dim=2)
        out = self.feature_extract(self.feature_extract_activation(out)).squeeze(2)
        return out
    def training_step(self, batch):
        x, y, seq_lens = batch
        y_norm, y_no_norm, y_agg_norm, y_agg_no_norm = y
        out = self.forward(x)
        # Masked loss; init mask here since module knows device
        mask = torch.zeros(max(seq_lens), len(seq_lens), dtype=torch.bool, device=self.device)
        mask = model_utils.fill_tensor_mask(mask, seq_lens)
        out = out[mask]
        labels = y_norm[mask]
        loss = self.loss_fn(out, labels)
        self.log_dict(
            {'train_loss': loss, 'batch_size': torch.tensor(y_agg_norm.shape[1], dtype=torch.float32)},
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=self.batch_size
        )
        return loss
    def validation_step(self, batch):
        x, y, seq_lens = batch
        y_norm, y_no_norm, y_agg_norm, y_agg_no_norm = y
        out = self.forward(x)
        # Masked loss; init mask here since module knows device
        mask = torch.zeros(max(seq_lens), len(seq_lens), dtype=torch.bool, device=self.device)
        mask = model_utils.fill_tensor_mask(mask, seq_lens)
        out = out[mask]
        labels = y_norm[mask]
        loss = self.loss_fn(out, labels)
        self.log_dict(
            {'valid_loss': loss, 'batch_size': torch.tensor(y_agg_norm.shape[1], dtype=torch.float32)},
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=self.batch_size
        )
        return loss
    def predict_step(self, batch):
        x, y, seq_lens = batch
        y_norm, y_no_norm, y_agg_norm, y_agg_no_norm = y
        out = self.forward(x)
        mask = torch.zeros(max(seq_lens), len(seq_lens), dtype=torch.bool, device=self.device)
        mask = model_utils.fill_tensor_mask(mask, seq_lens)
        out = data_loader.denormalize(out, self.config['calc_time_s'])
        out_agg = model_utils.aggregate_tts(out, mask)
        y_agg = model_utils.aggregate_tts(y_no_norm, mask)
        return {'preds': out_agg.detach().cpu().numpy(), 'labels': y_agg.detach().cpu().numpy(), 'preds_raw': out.detach().cpu().numpy(), 'labels_raw': y_no_norm.detach().cpu().numpy(), 'mask': mask.detach().cpu().numpy()}
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


class GRURealtime(pl.LightningModule):
    def __init__(self, model_name, config, holdout_routes, input_size, collate_fn, batch_size, hidden_size, num_layers, dropout_rate, grid_input_size, grid_compression_size):
        super(GRURealtime, self).__init__()
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
        self.include_grid = True
        self.loss_fn = torch.nn.MSELoss()
        self.grid_input_size = grid_input_size
        self.grid_compression_size = grid_compression_size
        # Embeddings
        self.min_em = embedding.MinuteEmbedding()
        self.day_em = embedding.DayEmbedding()
        self.embed_total_dims = self.min_em.embed_dim + self.day_em.embed_dim
        # Recurrent
        self.rnn = nn.GRU(input_size=self.input_size+self.grid_compression_size, hidden_size=self.hidden_size, num_layers=self.num_layers, dropout=self.dropout_rate)
        # Linear compression/feature extraction
        self.feature_extract = nn.Linear(in_features=self.hidden_size + self.embed_total_dims, out_features=1)
        self.feature_extract_activation = nn.ReLU()
        # Grid
        self.grid_stack = realtime.GridFeedForward(self.grid_input_size, self.grid_compression_size, self.hidden_size)
    def forward(self, x):
        x_em, x_ct, x_gr = x
        # Embed categorical variables
        x_min_em = self.min_em(x_em[:,:,0])
        x_day_em = self.day_em(x_em[:,:,1])
        # Grid
        x_gr = self.grid_stack(x_gr)
        x_ct = torch.cat([x_ct, x_gr], dim=2)
        # Get recurrent pred
        x_ct, hidden_prev = self.rnn(x_ct)
        # Combine all variables
        out = torch.cat([x_ct, x_min_em, x_day_em], dim=2)
        out = self.feature_extract(self.feature_extract_activation(out)).squeeze(2)
        return out
    def training_step(self, batch):
        x, y, seq_lens = batch
        y_norm, y_no_norm, y_agg_norm, y_agg_no_norm = y
        out = self.forward(x)
        # Masked loss; init mask here since module knows device
        mask = torch.zeros(max(seq_lens), len(seq_lens), dtype=torch.bool, device=self.device)
        mask = model_utils.fill_tensor_mask(mask, seq_lens)
        out = out[mask]
        labels = y_norm[mask]
        loss = self.loss_fn(out, labels)
        self.log_dict(
            {'train_loss': loss, 'batch_size': torch.tensor(y_agg_norm.shape[1], dtype=torch.float32)},
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=self.batch_size
        )
        return loss
    def validation_step(self, batch):
        x, y, seq_lens = batch
        y_norm, y_no_norm, y_agg_norm, y_agg_no_norm = y
        out = self.forward(x)
        # Masked loss; init mask here since module knows device
        mask = torch.zeros(max(seq_lens), len(seq_lens), dtype=torch.bool, device=self.device)
        mask = model_utils.fill_tensor_mask(mask, seq_lens)
        out = out[mask]
        labels = y_norm[mask]
        loss = self.loss_fn(out, labels)
        self.log_dict(
            {'valid_loss': loss, 'batch_size': torch.tensor(y_agg_norm.shape[1], dtype=torch.float32)},
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=self.batch_size
        )
        return loss
    def predict_step(self, batch):
        x, y, seq_lens = batch
        y_norm, y_no_norm, y_agg_norm, y_agg_no_norm = y
        out = self.forward(x)
        mask = torch.zeros(max(seq_lens), len(seq_lens), dtype=torch.bool, device=self.device)
        mask = model_utils.fill_tensor_mask(mask, seq_lens)
        out = data_loader.denormalize(out, self.config['calc_time_s'])
        out_agg = model_utils.aggregate_tts(out, mask)
        y_agg = model_utils.aggregate_tts(y_no_norm, mask)
        return {'preds': out_agg.detach().cpu().numpy(), 'labels': y_agg.detach().cpu().numpy(), 'preds_raw': out.detach().cpu().numpy(), 'labels_raw': y_no_norm.detach().cpu().numpy(), 'mask': mask.detach().cpu().numpy()}
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer