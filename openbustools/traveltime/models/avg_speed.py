import numpy as np
import pandas as pd

from openbustools.traveltime import data_loader


class AvgSpeedModel:
    def __init__(self, model_name, dataset, idx=None):
        self.model_name = model_name
        self.is_nn = False
        self.include_grid = False
        self.colnames = data_loader.LABEL_FEATS+data_loader.EMBED_FEATS+data_loader.GPS_FEATS+data_loader.STATIC_FEATS+data_loader.DEEPTTE_FEATS+data_loader.MISC_CON_FEATS
        if idx is None:
            idx = np.arange(len(dataset))
        speeds = np.concatenate([dataset.data[i]['feats_n'][:,self.colnames.index('calc_speed_m_s')] for i in idx])
        hours = np.concatenate([dataset.data[i]['feats_n'][:,self.colnames.index('t_hour')] for i in idx])
        mins = np.concatenate([dataset.data[i]['feats_n'][:,self.colnames.index('t_min_of_day')] for i in idx])
        self.speed_mean = np.mean(speeds)
        self.hour_speed_lookup = pd.DataFrame({'hours':hours, 'speeds':speeds}).groupby('hours').mean().to_dict()
        self.min_speed_lookup = pd.DataFrame({'mins':mins, 'speeds':speeds}).groupby('mins').mean().to_dict()
    def get_speed_if_available(self, h_or_m, t):
        if h_or_m=='h':
            if t in list(self.hour_speed_lookup.keys()):
                return self.hour_speed_lookup[t]
            else:
                return self.speed_mean
        elif h_or_m=='m':
            if t in list(self.min_speed_lookup.keys()):
                return self.min_speed_lookup[t]
            else:
                return self.speed_mean
    def predict(self, dataset, h_or_m):
        labels = np.array([dataset.data[x]['feats_n'][:,self.colnames.index('cumul_time_s')][-1] for x in np.arange(len(dataset))])
        dists = np.array([dataset.data[x]['feats_n'][:,self.colnames.index('cumul_dist_m')][-1] for x in np.arange(len(dataset))])
        if h_or_m=='h':
            times = [dataset.data[x]['feats_n'][:,self.colnames.index('t_hour')][0] for x in np.arange(len(dataset))]
            res = np.array([self.get_speed_if_available('h', x) for x in times])
        else:
            times = [dataset.data[x]['feats_n'][:,self.colnames.index('t_min_of_day')][0] for x in np.arange(len(dataset))]
            res = np.array([self.get_speed_if_available('m', x) for x in times])
        preds = dists / res
        return {'preds':preds, 'labels':labels}