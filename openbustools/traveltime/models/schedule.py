import numpy as np

from openbustools.traveltime import data_loader


class ScheduleModel:
    def __init__(self, model_name):
        self.model_name = model_name
        self.is_nn = False
        self.include_grid = False
        self.colnames = data_loader.LABEL_FEATS+data_loader.GPS_FEATS+data_loader.STATIC_FEATS+data_loader.DEEPTTE_FEATS+data_loader.MISC_CON_FEATS+data_loader.EMBED_FEATS
    def predict(self, dataset):
        labels = np.array([dataset.data[x]['feats_n'][:,self.colnames.index('cumul_time_s')][-1] for x in np.arange(len(dataset))])
        preds = np.array([dataset.data[x]['feats_n'][:,self.colnames.index('sch_time_s')][-1] for x in np.arange(len(dataset))])
        preds = np.clip(preds, 0, 3600)
        return {'preds':preds, 'labels':labels}