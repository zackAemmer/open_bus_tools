import numpy as np
import torch
from torch import nn
import lightning.pytorch as pl


class FF_L(pl.LightningModule):
    def __init__(self, model_name, n_features, hyperparameter_dict, embed_dict, collate_fn, config):
        super(FF_L, self).__init__()
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
        self.loss_fn = nn.HuberLoss()
        # Embeddings
        self.embed_total_dims = np.sum([self.embed_dict[key]['embed_dims'] for key in self.embed_dict.keys()]).astype('int32')
        self.timeID_em = nn.Embedding(self.embed_dict['timeID']['vocab_size'], self.embed_dict['timeID']['embed_dims'])
        self.weekID_em = nn.Embedding(self.embed_dict['weekID']['vocab_size'], self.embed_dict['weekID']['embed_dims'])
        # Feedforward
        self.linear_relu_stack = nn.Sequential()
        self.linear_relu_stack.append(nn.BatchNorm1d(self.n_features + self.embed_total_dims))
        self.linear_relu_stack.append(nn.Linear(self.n_features + self.embed_total_dims, self.hyperparameter_dict['hidden_size']))
        self.linear_relu_stack.append(nn.ReLU())
        for i in range(self.hyperparameter_dict['num_layers']):
            self.linear_relu_stack.append(nn.Linear(self.hyperparameter_dict['hidden_size'], self.hyperparameter_dict['hidden_size']))
            self.linear_relu_stack.append(nn.ReLU())
        self.linear_relu_stack.append(nn.Dropout(p=self.hyperparameter_dict['dropout_rate']))
        self.feature_extract = nn.Linear(self.hyperparameter_dict['hidden_size'], 1)
        self.feature_extract_activation = nn.ReLU()
    def training_step(self, batch, batch_idx):
        x,y = batch
        x_em = x[0]
        x_ct = x[1]
        timeID_embedded = self.timeID_em(x_em[:,0])
        weekID_embedded = self.weekID_em(x_em[:,1])
        out = torch.cat([x_ct, timeID_embedded, weekID_embedded], dim=1)
        out = self.linear_relu_stack(out)
        out = self.feature_extract(self.feature_extract_activation(out)).squeeze()
        loss = self.loss_fn(out, y)
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
        timeID_embedded = self.timeID_em(x_em[:,0])
        weekID_embedded = self.weekID_em(x_em[:,1])
        out = torch.cat([x_ct, timeID_embedded, weekID_embedded], dim=1)
        out = self.linear_relu_stack(out)
        out = self.feature_extract(self.feature_extract_activation(out)).squeeze()
        loss = self.loss_fn(out, y)
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
        timeID_embedded = self.timeID_em(x_em[:,0])
        weekID_embedded = self.weekID_em(x_em[:,1])
        out = torch.cat([x_ct, timeID_embedded, weekID_embedded], dim=1)
        out = self.linear_relu_stack(out)
        out = self.feature_extract(self.feature_extract_activation(out)).squeeze()
        out  = (out.detach().cpu().numpy() * self.config['time_std']) + self.config['time_mean']
        y = (y.detach().cpu().numpy() * self.config['time_std']) + self.config['time_mean']
        return {"out_agg":out, "y_agg":y}
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

class FF_GRID_L(pl.LightningModule):
    def __init__(self, model_name, n_features, n_grid_features, grid_compression_size, hyperparameter_dict, embed_dict, collate_fn, config):
        super(FF_GRID_L, self).__init__()
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
        self.loss_fn = nn.HuberLoss()
        # Embeddings
        self.embed_total_dims = np.sum([self.embed_dict[key]['embed_dims'] for key in self.embed_dict.keys()]).astype('int32')
        self.timeID_em = nn.Embedding(self.embed_dict['timeID']['vocab_size'], self.embed_dict['timeID']['embed_dims'])
        self.weekID_em = nn.Embedding(self.embed_dict['weekID']['vocab_size'], self.embed_dict['weekID']['embed_dims'])
        # Grid Feedforward
        self.linear_relu_stack_grid = nn.Sequential(
            nn.BatchNorm1d(self.n_grid_features),
            nn.Linear(self.n_grid_features, self.hyperparameter_dict['hidden_size']),
            nn.ReLU(),
            nn.Linear(self.hyperparameter_dict['hidden_size'], self.grid_compression_size),
            nn.ReLU()
        )
        # Feedforward
        self.linear_relu_stack = nn.Sequential()
        self.linear_relu_stack.append(nn.BatchNorm1d(self.n_features + self.embed_total_dims + self.grid_compression_size))
        self.linear_relu_stack.append(nn.Linear(self.n_features + self.embed_total_dims + self.grid_compression_size, self.hyperparameter_dict['hidden_size']))
        self.linear_relu_stack.append(nn.ReLU())
        for i in range(self.hyperparameter_dict['num_layers']):
            self.linear_relu_stack.append(nn.Linear(self.hyperparameter_dict['hidden_size'], self.hyperparameter_dict['hidden_size']))
            self.linear_relu_stack.append(nn.ReLU())
        self.linear_relu_stack.append(nn.Dropout(p=self.hyperparameter_dict['dropout_rate']))
        self.feature_extract = nn.Linear(self.hyperparameter_dict['hidden_size'], 1)
        self.feature_extract_activation = nn.ReLU()
    def training_step(self, batch, batch_idx):
        x,y = batch
        x_em = x[0]
        x_ct = x[1]
        x_gr = x[2]
        # Embed categorical variables
        timeID_embedded = self.timeID_em(x_em[:,0])
        weekID_embedded = self.weekID_em(x_em[:,1])
        # Feed grid data through model
        x_gr = self.linear_relu_stack_grid(torch.flatten(x_gr, 1))
        # Feed data through the model
        out = torch.cat([x_gr, x_ct, timeID_embedded, weekID_embedded], dim=1)
        # Make prediction
        out = self.linear_relu_stack(out)
        out = self.feature_extract(self.feature_extract_activation(out)).squeeze()
        # Get loss
        loss = self.loss_fn(out, y)
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
        # Embed categorical variables
        timeID_embedded = self.timeID_em(x_em[:,0])
        weekID_embedded = self.weekID_em(x_em[:,1])
        # Feed grid data through model
        x_gr = self.linear_relu_stack_grid(torch.flatten(x_gr, 1))
        # Feed data through the model
        out = torch.cat([x_gr, x_ct, timeID_embedded, weekID_embedded], dim=1)
        # Make prediction
        out = self.linear_relu_stack(out)
        out = self.feature_extract(self.feature_extract_activation(out)).squeeze()
        # Get loss
        loss = self.loss_fn(out, y)
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
        # Embed categorical variables
        timeID_embedded = self.timeID_em(x_em[:,0])
        weekID_embedded = self.weekID_em(x_em[:,1])
        # Feed grid data through model
        x_gr = self.linear_relu_stack_grid(torch.flatten(x_gr, 1))
        # Feed data through the model
        out = torch.cat([x_gr, x_ct, timeID_embedded, weekID_embedded], dim=1)
        # Make prediction
        out = self.linear_relu_stack(out)
        out = self.feature_extract(self.feature_extract_activation(out)).squeeze()
        out  = (out.detach().cpu().numpy() * self.config['time_std']) + self.config['time_mean']
        y = (y.detach().cpu().numpy() * self.config['time_std']) + self.config['time_mean']
        return {"out_agg":out, "y_agg":y}
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer