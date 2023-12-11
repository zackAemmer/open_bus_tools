import numpy as np
import pandas as pd

from openbustools.traveltime import data_loader


class AvgSpeedModel:
    """
    A class representing an average speed model.

    Attributes:
        model_name (str): The name of the model.
        is_nn (bool): Indicates whether the model is a neural network model.
        include_grid (bool): Indicates whether the model includes grid information.
        colnames (list): The column names of the dataset.
        speed_mean (float): The mean speed of the dataset.
        hour_speed_lookup (dict): A dictionary mapping hours to average speeds.
        min_speed_lookup (dict): A dictionary mapping minutes to average speeds.

    Methods:
        __init__(self, model_name, dataset, idx=None): Initializes the AvgSpeedModel object.
        predict(self, dataset, h_or_m): Predicts the travel time based on the dataset and the hour or minute of the day.
    """

    def __init__(self, model_name, dataset, idx=None):
        self.model_name = model_name
        self.is_nn = False
        self.include_grid = False
        self.colnames = data_loader.NUM_FEAT_COLS
        if idx is None:
            idx = np.arange(len(dataset))
        speeds = np.concatenate([dataset.data[i]['feats_n'][:,self.colnames.index('calc_speed_m_s')] for i in idx]).clip(1, 40)
        hours = np.concatenate([dataset.data[i]['feats_n'][:,self.colnames.index('t_hour')] for i in idx])
        mins = np.concatenate([dataset.data[i]['feats_n'][:,self.colnames.index('t_min_of_day')] for i in idx])
        self.speed_mean = np.mean(speeds)
        self.hour_speed_lookup = pd.DataFrame({'hours':hours, 'speeds':speeds}).groupby('hours').mean().to_dict()
        self.min_speed_lookup = pd.DataFrame({'mins':mins, 'speeds':speeds}).groupby('mins').mean().to_dict()

    def predict(self, dataset, h_or_m):
        """
        Predicts the travel time based on the given dataset and time granularity.

        Args:
            dataset (Dataset): The dataset containing the input features.
            h_or_m (str): The time granularity, either 'h' for hour or 'm' for minute.

        Returns:
            list: A list of dictionaries containing the predicted travel time, actual labels,
                  raw predicted values, and raw label values for each data point in the dataset.
        """
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
            times_df = pd.DataFrame({'mins': times})
            speeds_df = pd.merge(times_df, speeds_lookup_df, left_on='mins', right_on='index', how='left')
            speeds_df['speeds'] = speeds_df['speeds'].fillna(self.speed_mean)
            speeds = speeds_df['speeds'].to_numpy()
        preds_raw = [dists / speed for dists, speed in zip(dists_raw, speeds)]
        preds = dists / speeds
        # Distance of 0 is a special case, cannot predict time so guess average of other times in the sequence
        preds_mean = [np.mean(i) for i in preds_raw]
        for i, pred in enumerate(preds_raw):
            if np.sum(pred==0) > 0:
                pred[pred==0] = preds_mean[i]
        res = [{'preds': preds[i], 'labels': labels[i], 'preds_raw': preds_raw[i], 'labels_raw': labels_raw[i]} for i in range(len(preds))]
        return res