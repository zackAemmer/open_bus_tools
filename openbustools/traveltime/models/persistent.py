import numpy as np

from openbustools.traveltime import data_loader


class PersistentTimeModel:
    def __init__(self, model_name):
        self.model_name = model_name
        self.is_nn = False
        self.include_grid = False
        self.colnames = data_loader.NUM_FEAT_COLS
    def predict(self, dataset, idx=None):
        if idx is None:
            idx = np.arange(len(dataset))
        preds = []
        labels = []
        for i in idx:
            sample = dataset.find_sample(i)
            preds.append(np.sum(np.repeat(np.array(30), sample.shape[0] - 1)))
            labels.append(sample[:,self.colnames.index('cumul_time_s')][-1])
        res = {'preds': np.array(preds), 'labels': np.array(labels)}
        return res