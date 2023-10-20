import numpy as np
import pandas as pd
import lightning.pytorch as pl

from obt import data_loader
from obt import data_utils


class AvgHourlySpeedModel:
    def __init__(self, model_name):
        self.model_name = model_name
        self.speed_lookup = {}
        self.requires_grid = False
        self.collate_fn = data_loader.avg_collate
        self.train_time = 0.0
        self.batch_size = 512
        self.is_nn = False
        return None
    def train(self, dataloader, config):
        speeds = []
        dists = []
        hours = []
        for i in dataloader:
            speeds.extend(i[0])
            dists.extend(i[1])
            hours.extend(i[2])
        # Calculate average speed grouped by time of day
        self.speed_lookup = pd.DataFrame({"hour":hours, "speed":speeds}).groupby("hour").mean().to_dict()
        return None
    def evaluate(self, dataloader, config):
        speeds = []
        dists = []
        hours = []
        labels = []
        for i in dataloader:
            speeds.extend(i[0])
            dists.extend(i[1])
            hours.extend(i[2])
            labels.extend(i[3])
        pred_speeds = np.array([self.get_speed_if_available(x) for x in hours])
        preds = list(dists / pred_speeds)
        return np.array(labels), np.array(preds)
    def get_speed_if_available(self, hour):
        # If no data was available for the requested hour, return the mean of all available hours
        if hour in self.speed_lookup['speed'].keys():
            speed = self.speed_lookup['speed'][hour]
            # If there is an hour with 0.0 speeds due to small sample size it will cause errors
            if speed == 0.0:
                return np.mean(list(self.speed_lookup['speed'].values()))
            else:
                return speed
        else:
            return np.mean(list(self.speed_lookup['speed'].values()))
    def save_to(self, path):
        data_utils.write_pkl(self, path)
        return None