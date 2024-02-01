import numpy as np

from openbustools.traveltime import data_loader


class ScheduleModel:
    def __init__(self, model_name):
        self.model_name = model_name
        self.is_nn = False
        self.include_grid = False
        self.colnames = data_loader.LABEL_FEATS+data_loader.GPS_FEATS+data_loader.STATIC_FEATS+data_loader.DEEPTTE_FEATS+data_loader.MISC_CON_FEATS+data_loader.EMBED_FEATS
    def predict(self, dataset, idx=None):
        if idx is None:
            idx = np.arange(len(dataset))
        preds = []
        labels = []
        for i in idx:
            sample = dataset.find_sample(i)
            preds.append(sample[:,self.colnames.index('sch_time_s')][-1])
            labels.append(sample[:,self.colnames.index('cumul_time_s')][-1])
        # preds = np.clip(preds, 0, 3600)
        res = {'preds': np.array(preds), 'labels': np.array(labels)}
        return res