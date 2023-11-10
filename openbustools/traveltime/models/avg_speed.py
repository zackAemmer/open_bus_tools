import numpy as np
import pandas as pd
import lightning.pytorch as pl

from openbustools.traveltime import data_loader
from openbustools import data_utils


class AvgSpeedModel:
    def __init__(self, model_name, data_df):
        self.model_name = model_name
        self.speed_mean = np.mean(data_df['calc_speed_m_s'].to_numpy())
        self.hour_speed_lookup = pd.DataFrame({'hour':data_df['t_hour'], 'speed':data_df['calc_speed_m_s']}).groupby('hour').mean().to_dict()
        self.min_speed_lookup = pd.DataFrame({'min':data_df['t_min'], 'speed':data_df['calc_speed_m_s']}).groupby('min').mean().to_dict()
        self.is_nn = False
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