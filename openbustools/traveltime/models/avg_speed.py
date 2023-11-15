import numpy as np
import pandas as pd


class AvgSpeedModel:
    def __init__(self, model_name, data_df):
        self.model_name = model_name
        self.is_nn = False
        self.include_grid = False
        self.speed_mean = np.mean(data_df['calc_speed_m_s'].to_numpy())
        self.hour_speed_lookup = pd.DataFrame({'hour':data_df['t_hour'], 'speed':data_df['calc_speed_m_s']}).groupby('hour').mean().to_dict()
        self.min_speed_lookup = pd.DataFrame({'min':data_df['t_min'], 'speed':data_df['calc_speed_m_s']}).groupby('min').mean().to_dict()
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
        data_df = dataset.data
        res = data_df.groupby('shingle_id')[['cumul_dist_km','t_hour','t_min','cumul_time_s']].last()
        if h_or_m=='h':
            res['speeds'] = [self.get_speed_if_available('h', x) for x in res['t_hour']]
        else:
            res['speeds'] = [self.get_speed_if_available('m', x) for x in res['t_min']]
        res['preds'] = res['cumul_dist_km']*1000 / res['speeds']
        res['labels'] = res['cumul_time_s']
        return {'preds':res['preds'].to_numpy(), 'labels':res['labels'].to_numpy()}