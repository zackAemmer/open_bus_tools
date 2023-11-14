import numpy as np

from openbustools.traveltime import data_loader
from openbustools import data_utils


class ScheduleModel:
    def __init__(self, model_name):
        self.model_name = model_name
        self.is_nn = False
    def predict(self, dataset):
        data_df = dataset.data
        res = data_df.groupby('shingle_id')[['sch_time_s','cumul_time_s']].last()
        res['preds'] = res['sch_time_s']
        res['labels'] = res['cumul_time_s']
        return {'preds':res['preds'].to_numpy(), 'labels':res['labels'].to_numpy()}