import numpy as np
import pandas as pd

from openbustools.traveltime import data_loader


class AvgSpeedModel:
    def __init__(self, model_name, dataset, idx=None):
        self.model_name = model_name
        self.is_nn = False
        self.include_grid = False
        self.colnames = data_loader.LABEL_FEATS+data_loader.GPS_FEATS+data_loader.STATIC_FEATS+data_loader.DEEPTTE_FEATS+data_loader.MISC_CON_FEATS+data_loader.EMBED_FEATS
        if idx is None:
            idx = np.arange(len(dataset))
        speeds = np.concatenate([dataset.data[i]['feats_n'][:,self.colnames.index('calc_speed_m_s')] for i in idx])
        hours = np.concatenate([dataset.data[i]['feats_n'][:,self.colnames.index('t_hour')] for i in idx])
        mins = np.concatenate([dataset.data[i]['feats_n'][:,self.colnames.index('t_min_of_day')] for i in idx])
        self.speed_mean = np.mean(speeds)
        self.hour_speed_lookup = pd.DataFrame({'hours':hours, 'speeds':speeds}).groupby('hours').mean().to_dict()
        self.min_speed_lookup = pd.DataFrame({'mins':mins, 'speeds':speeds}).groupby('mins').mean().to_dict()
    def predict(self, dataset, h_or_m):
        labels_raw = [dataset.data[x]['feats_n'][:,self.colnames.index('calc_time_s')] for x in np.arange(len(dataset))]
        dists_raw = [dataset.data[x]['feats_n'][:,self.colnames.index('calc_dist_m')] for x in np.arange(len(dataset))]
        labels = np.array([dataset.data[x]['feats_n'][:,self.colnames.index('cumul_time_s')][-1] for x in np.arange(len(dataset))])
        dists = np.array([dataset.data[x]['feats_n'][:,self.colnames.index('cumul_dist_m')][-1] for x in np.arange(len(dataset))])
        if h_or_m=='h':
            speeds_lookup_df = pd.DataFrame(self.hour_speed_lookup).reset_index()
            times = [dataset.data[x]['feats_n'][:,self.colnames.index('t_hour')][0] for x in np.arange(len(dataset))]
            times_df = pd.DataFrame({'hours':times})
            speeds_df = pd.merge(times_df, speeds_lookup_df, left_on='hours', right_on='index', how='left')
            speeds_df['speeds'] = speeds_df['speeds'].fillna(self.speed_mean)
            speeds = speeds_df['speeds'].to_numpy()
        else:
            times = [dataset.data[x]['feats_n'][:,self.colnames.index('t_min_of_day')][0] for x in np.arange(len(dataset))]
            speeds_lookup_df = pd.DataFrame(self.min_speed_lookup).reset_index()
            times_df = pd.DataFrame({'mins':times})
            speeds_df = pd.merge(times_df, speeds_lookup_df, left_on='mins', right_on='index', how='left')
            speeds_df['speeds'] = speeds_df['speeds'].fillna(self.speed_mean)
            speeds = speeds_df['speeds'].to_numpy()
        preds_raw = [dists_raw[i] / speeds[i] for i in range(len(dists_raw))]
        preds = dists / speeds
        return {'preds':preds, 'labels':labels, 'preds_raw':preds_raw, 'labels_raw':labels_raw}