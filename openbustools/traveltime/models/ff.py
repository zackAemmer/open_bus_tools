import numpy as np
import torch
import lightning.pytorch as pl

from openbustools.traveltime import data_loader, model_utils
from openbustools.traveltime.models import embedding, realtime


class FF(pl.LightningModule):
    def __init__(self, model_name, config, holdout_routes, input_size, collate_fn, batch_size, hidden_size, num_layers, dropout_rate):
        super(FF, self).__init__()
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
        # Feedforward
        self.linear_relu_stack = torch.nn.Sequential()
        self.linear_relu_stack.append(torch.nn.Linear(2 * (self.input_size+self.embed_total_dims), self.hidden_size))
        self.linear_relu_stack.append(torch.nn.ReLU())
        for _ in range(self.num_layers):
            self.linear_relu_stack.append(torch.nn.Linear(self.hidden_size, self.hidden_size))
            self.linear_relu_stack.append(torch.nn.ReLU())
        self.linear_relu_stack.append(torch.nn.Dropout(p=self.dropout_rate))
        # Linear compression/feature extraction
        self.feature_extract = torch.nn.Linear(self.hidden_size, 1)
        self.feature_extract_activation = torch.nn.ReLU()
    def forward(self, x, seq_lens):
        x_em, x_ct = x
        # Embed categorical variables
        x_min_em = self.min_em(x_em[:,:,0])
        x_day_em = self.day_em(x_em[:,:,1])
        # Run through feedforward
        out = torch.cat([x_ct, x_min_em, x_day_em], dim=2)
        idxs = seq_lens.unsqueeze(0).unsqueeze(-1).expand(1, out.shape[1], out.shape[2]) - 1
        first = out[0,:,:].squeeze(0)
        last = torch.gather(out, 0, idxs).squeeze(0)
        out = self.linear_relu_stack(torch.cat([first, last], dim=1))
        out = self.feature_extract(self.feature_extract_activation(out)).squeeze()
        return out
    def training_step(self, batch):
        x, y, seq_lens = batch
        y_norm, y_no_norm, y_agg_norm, y_agg_no_norm = y
        out = self.forward(x, seq_lens)
        idxs = seq_lens.unsqueeze(0) - 1
        labels = torch.gather(y_agg_norm, 0, idxs).squeeze()
        loss = self.loss_fn(out, labels)
        self.log_dict(
            {'train_loss': loss},
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return loss
    def validation_step(self, batch):
        x, y, seq_lens = batch
        y_norm, y_no_norm, y_agg_norm, y_agg_no_norm = y
        out = self.forward(x, seq_lens)
        idxs = seq_lens.unsqueeze(0) - 1
        labels = torch.gather(y_agg_norm, 0, idxs).squeeze()
        loss = self.loss_fn(out, labels)
        self.log_dict(
            {'valid_loss': loss},
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return loss
    def predict_step(self, batch):
        x, y, seq_lens = batch
        y_norm, y_no_norm, y_agg_norm, y_agg_no_norm = y
        out = self.forward(x, seq_lens)
        out = data_loader.denormalize(out, self.config['cumul_time_s'])
        idxs = seq_lens.unsqueeze(0) - 1
        labels = torch.gather(y_agg_no_norm, 0, idxs).squeeze()
        return {'preds': out.detach().cpu().numpy(), 'labels': labels.detach().cpu().numpy()}
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


class FFRealtime(pl.LightningModule):
    def __init__(self, model_name, config, holdout_routes, input_size, collate_fn, batch_size, hidden_size, num_layers, dropout_rate, grid_input_size, grid_compression_size):
        super(FFRealtime, self).__init__()
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
        # Feedforward
        self.linear_relu_stack = torch.nn.Sequential()
        self.linear_relu_stack.append(torch.nn.Linear(2 * (self.input_size+self.embed_total_dims+self.grid_compression_size), self.hidden_size))
        self.linear_relu_stack.append(torch.nn.ReLU())
        for _ in range(self.num_layers):
            self.linear_relu_stack.append(torch.nn.Linear(self.hidden_size, self.hidden_size))
            self.linear_relu_stack.append(torch.nn.ReLU())
        self.linear_relu_stack.append(torch.nn.Dropout(p=self.dropout_rate))
        # Linear compression/feature extraction
        self.feature_extract = torch.nn.Linear(self.hidden_size, 1)
        self.feature_extract_activation = torch.nn.ReLU()
        # Grid
        self.grid_stack = realtime.GridFeedForward(self.grid_input_size, self.grid_compression_size, self.hidden_size)
    def forward(self, x, seq_lens):
        x_em, x_ct, x_gr = x
        # Embed categorical variables
        x_min_em = self.min_em(x_em[:,:,0])
        x_day_em = self.day_em(x_em[:,:,1])
        # Grid
        x_gr = self.grid_stack(x_gr)
        # Run through feedforward
        out = torch.cat([x_ct, x_gr, x_min_em, x_day_em], dim=2)
        idxs = seq_lens.unsqueeze(0).unsqueeze(-1).expand(1, out.shape[1], out.shape[2]) - 1
        first = out[0,:,:].squeeze(0)
        last = torch.gather(out, 0, idxs).squeeze(0)
        out = self.linear_relu_stack(torch.cat([first, last], dim=1))
        out = self.feature_extract(self.feature_extract_activation(out)).squeeze()
        return out
    def training_step(self, batch):
        x, y, seq_lens = batch
        y_norm, y_no_norm, y_agg_norm, y_agg_no_norm = y
        out = self.forward(x, seq_lens)
        idxs = seq_lens.unsqueeze(0) - 1
        labels = torch.gather(y_agg_norm, 0, idxs).squeeze()
        loss = self.loss_fn(out, labels)
        self.log_dict(
            {'train_loss': loss},
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return loss
    def validation_step(self, batch):
        x, y, seq_lens = batch
        y_norm, y_no_norm, y_agg_norm, y_agg_no_norm = y
        out = self.forward(x, seq_lens)
        idxs = seq_lens.unsqueeze(0) - 1
        labels = torch.gather(y_agg_norm, 0, idxs).squeeze()
        loss = self.loss_fn(out, labels)
        self.log_dict(
            {'valid_loss': loss},
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return loss
    def predict_step(self, batch):
        x, y, seq_lens = batch
        y_norm, y_no_norm, y_agg_norm, y_agg_no_norm = y
        out = self.forward(x, seq_lens)
        out = data_loader.denormalize(out, self.config['cumul_time_s'])
        idxs = seq_lens.unsqueeze(0) - 1
        labels = torch.gather(y_agg_no_norm, 0, idxs).squeeze()
        return {'preds': out.detach().cpu().numpy(), 'labels': labels.detach().cpu().numpy()}
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer