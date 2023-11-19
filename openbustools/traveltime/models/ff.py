import numpy as np
import torch
import lightning.pytorch as pl

from openbustools.traveltime import data_loader
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
        self.loss_fn = torch.nn.HuberLoss()
        # Embeddings
        self.min_em = embedding.MinuteEmbedding()
        self.day_em = embedding.DayEmbedding()
        self.embed_total_dims = self.min_em.embed_dim + self.day_em.embed_dim
        # Feedforward
        self.linear_relu_stack = torch.nn.Sequential()
        self.linear_relu_stack.append(torch.nn.Linear(self.input_size+self.embed_total_dims, self.hidden_size))
        self.linear_relu_stack.append(torch.nn.ReLU())
        for _ in range(self.num_layers):
            self.linear_relu_stack.append(torch.nn.Linear(self.hidden_size, self.hidden_size))
            self.linear_relu_stack.append(torch.nn.ReLU())
        self.linear_relu_stack.append(torch.nn.Dropout(p=self.dropout_rate))
        # Linear compression/feature extraction
        self.feature_extract = torch.nn.Linear(self.hidden_size, 1)
        self.feature_extract_activation = torch.nn.ReLU()
    def forward(self, x_em, x_ct):
        # Embed categorical variables
        x_min_em = self.min_em(x_em[:,0])
        x_day_em = self.day_em(x_em[:,1])
        # Combine all variables
        out = torch.cat([x_ct, x_min_em, x_day_em], dim=1)
        out = self.linear_relu_stack(out)
        out = self.feature_extract(self.feature_extract_activation(out)).squeeze()
        return out
    def training_step(self, batch):
        x,y = batch
        x_em = x[0]
        x_ct = x[1]
        y_norm = y[0]
        y_no_norm = y[1]
        out = self.forward(x_em, x_ct)
        loss = self.loss_fn(out, y_norm)
        self.log_dict(
            {'train_loss': loss},
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return loss
    def validation_step(self, batch):
        x,y = batch
        x_em = x[0]
        x_ct = x[1]
        y_norm = y[0]
        y_no_norm = y[1]
        out = self.forward(x_em, x_ct)
        loss = self.loss_fn(out, y_norm)
        self.log_dict(
            {'valid_loss': loss},
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return loss
    def predict_step(self, batch):
        x,y = batch
        x_em = x[0]
        x_ct = x[1]
        y_norm = y[0]
        y_no_norm = y[1]
        out = self.forward(x_em, x_ct)
        out = out.detach().cpu().numpy()
        out = data_loader.denormalize(out, self.config['cumul_time_s'])
        y_no_norm = y_no_norm.detach().cpu().numpy()
        return {'preds':out, 'labels':y_no_norm}
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
        self.loss_fn = torch.nn.HuberLoss()
        self.grid_input_size = grid_input_size
        self.grid_compression_size = grid_compression_size
        # Embeddings
        self.min_em = embedding.MinuteEmbedding()
        self.day_em = embedding.DayEmbedding()
        self.embed_total_dims = self.min_em.embed_dim + self.day_em.embed_dim
        # Feedforward
        self.linear_relu_stack = torch.nn.Sequential()
        self.linear_relu_stack.append(torch.nn.Linear(self.input_size+self.embed_total_dims+(2*self.grid_compression_size), self.hidden_size))
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
    def forward(self, x_em, x_ct, x_gr):
        # Embed categorical variables
        x_min_em = self.min_em(x_em[:,0])
        x_day_em = self.day_em(x_em[:,1])
        # Grid
        x_gr = self.grid_stack(x_gr)
        x_gr = x_gr.view(x_gr.shape[0], x_gr.shape[1]*x_gr.shape[2])
        # Combine all variables
        out = torch.cat([x_ct, x_min_em, x_day_em, x_gr], dim=1)
        out = self.linear_relu_stack(out)
        out = self.feature_extract(self.feature_extract_activation(out)).squeeze()
        return out
    def training_step(self, batch):
        x,y = batch
        x_em = x[0]
        x_ct = x[1]
        x_gr = x[2]
        y_norm = y[0]
        y_no_norm = y[1]
        out = self.forward(x_em, x_ct, x_gr)
        loss = self.loss_fn(out, y_norm)
        self.log_dict(
            {'train_loss': loss},
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return loss
    def validation_step(self, batch):
        x,y = batch
        x_em = x[0]
        x_ct = x[1]
        x_gr = x[2]
        y_norm = y[0]
        y_no_norm = y[1]
        out = self.forward(x_em, x_ct, x_gr)
        loss = self.loss_fn(out, y_norm)
        self.log_dict(
            {'valid_loss': loss},
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return loss
    def predict_step(self, batch):
        x,y = batch
        x_em = x[0]
        x_ct = x[1]
        x_gr = x[2]
        y_norm = y[0]
        y_no_norm = y[1]
        out = self.forward(x_em, x_ct, x_gr)
        out = out.detach().cpu().numpy()
        out = data_loader.denormalize(out, self.config['cumul_time_s'])
        y_no_norm = y_no_norm.detach().cpu().numpy()
        return {'preds':out, 'labels':y_no_norm}
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer