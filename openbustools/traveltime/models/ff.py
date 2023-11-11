import numpy as np
import torch
import lightning.pytorch as pl

from openbustools.traveltime.models import embedding


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
    def training_step(self, batch):
        x,y = batch
        x_em = x[0]
        x_ct = x[1]
        # Embed categorical variables
        x_min_em = self.min_em(x_em[:,0])
        x_day_em = self.day_em(x_em[:,1])
        # Combine all variables
        out = torch.cat([x_ct, x_min_em, x_day_em], dim=1)
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
        return loss
    def validation_step(self, batch):
        x,y = batch
        x_em = x[0]
        x_ct = x[1]
        # Embed categorical variables
        x_min_em = self.min_em(x_em[:,0])
        x_day_em = self.day_em(x_em[:,1])
        # Combine all variables
        out = torch.cat([x_ct, x_min_em, x_day_em], dim=1)
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
    def predict_step(self, batch):
        x,y = batch
        x_em = x[0]
        x_ct = x[1]
        # Embed categorical variables
        x_min_em = self.min_em(x_em[:,0])
        x_day_em = self.day_em(x_em[:,1])
        # Combine all variables
        out = torch.cat([x_ct, x_min_em, x_day_em], dim=1)
        out = self.linear_relu_stack(out)
        out = self.feature_extract(self.feature_extract_activation(out)).squeeze()
        return {"preds":out, "labels":y}
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


class FF_REALTIME(pl.LightningModule):
    def __init__(self, model_name, config, holdout_routes, input_size, collate_fn, batch_size, hidden_size, num_layers, dropout_rate):
        super(FF_REALTIME, self).__init__()
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
        self.loss_fn = torch.nn.HuberLoss()
        # Embeddings
        self.min_em = embedding.MinuteEmbedding()
        self.day_em = embedding.DayEmbedding()
        self.embed_total_dims = self.min_em.embed_dim + self.day_em.embed_dim
        # Grid Feedforward
        self.linear_relu_stack_grid = torch.nn.Sequential(
            torch.nn.BatchNorm1d(self.n_grid_features),
            torch.nn.Linear(self.n_grid_features, self.hyperparameter_dict['hidden_size']),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hyperparameter_dict['hidden_size'], self.grid_compression_size),
            torch.nn.ReLU()
        )
        # Feedforward
        self.linear_relu_stack = torch.nn.Sequential()
        self.linear_relu_stack.append(torch.nn.BatchNorm1d(self.n_features + self.embed_total_dims + self.grid_compression_size))
        self.linear_relu_stack.append(torch.nn.Linear(self.n_features + self.embed_total_dims + self.grid_compression_size, self.hyperparameter_dict['hidden_size']))
        self.linear_relu_stack.append(torch.nn.ReLU())
        for i in range(self.hyperparameter_dict['num_layers']):
            self.linear_relu_stack.append(torch.nn.Linear(self.hyperparameter_dict['hidden_size'], self.hyperparameter_dict['hidden_size']))
            self.linear_relu_stack.append(torch.nn.ReLU())
        self.linear_relu_stack.append(torch.nn.Dropout(p=self.hyperparameter_dict['dropout_rate']))
        self.feature_extract = torch.nn.Linear(self.hyperparameter_dict['hidden_size'], 1)
        self.feature_extract_activation = torch.nn.ReLU()
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