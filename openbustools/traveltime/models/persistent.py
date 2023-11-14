import numpy as np


class PersistentTimeModel:
    def __init__(self, model_name):
        self.model_name = model_name
        self.is_nn = False
    def predict(self, dataset):
        data_df = dataset.data
        res = data_df.groupby('shingle_id')[['cumul_dist_km']].count()
        res['preds'] = (res['cumul_dist_km'] - 1) * 30
        res['labels'] = data_df.groupby('shingle_id')[['cumul_time_s']].last()
        return {'preds':res['preds'].to_numpy(), 'labels':res['labels'].to_numpy()}