import numpy as np

from openbustools.traveltime import data_loader


class AvgSpeedModel:
    """
    A class representing an average speed model.

    Attributes:
        model_name (str): The name of the model.
        config (dict): The configuration dictionary.
        holdout_routes (list): A list of routes to hold out from training.
        is_nn (bool): Indicates whether the model is a neural network model.
        include_grid (bool): Indicates whether the model includes grid information.
        colnames (list): The column names of the dataset.
        hour_speed_lookup (dict): A dictionary mapping hours to average speeds.

    Methods:
        __init__(self, model_name, dataset, config, holdout_routes, idx=None): Initializes the AvgSpeedModel object.
        predict(self, dataset): Predicts the travel time.
    """
    def __init__(self, model_name, dataset, idx=None, config=None, holdout_routes=None):
        self.model_name = model_name
        self.config = config
        self.holdout_routes = holdout_routes
        self.is_nn = False
        self.include_grid = False
        self.colnames = data_loader.NUM_FEAT_COLS
        if idx is None:
            idx = np.arange(len(dataset))
        # Running total of speed and count of samples for each hour
        self.hour_speed_lookup = {i: (0,0) for i in range(24)}
        for i in idx:
            sample = dataset.find_sample(i)
            sample_speed = np.mean(sample[:,self.colnames.index('calc_speed_m_s')])
            sample_time = sample[:,self.colnames.index('t_hour')][0]
            self.hour_speed_lookup[sample_time] = (self.hour_speed_lookup[sample_time][0] + sample_speed, self.hour_speed_lookup[sample_time][1] + 1)
        # Handle case where no samples for a given hour
        for i in range(24):
            if self.hour_speed_lookup[i][1] == 0:
                self.hour_speed_lookup[i] = (6.0,1)
        # Calculate average speed for each hour
        self.hour_speed_lookup = {i: self.hour_speed_lookup[i][0] / self.hour_speed_lookup[i][1] for i in range(24)}
    def predict(self, dataset, idx=None):
        """
        Predicts the travel time based on the given dataset and historic hourly speeds.

        Args:
            dataset (Dataset): The dataset containing the input features.
            idx (list): A list of indices to use for prediction. Defaults to None.

        Returns:
            list: A list of dictionaries containing the predicted travel time, actual labels,
                  raw predicted values, and raw label values for each data point in the dataset.
        """
        if idx is None:
            idx = np.arange(len(dataset))
        preds = []
        labels = []
        for i in idx:
            sample = dataset.find_sample(i)
            sample_dist = np.mean(sample[:,self.colnames.index('cumul_dist_m')][-1])
            sample_time = sample[:,self.colnames.index('t_hour')][0]
            pred_speed = self.hour_speed_lookup[sample_time]
            pred_time = sample_dist / pred_speed
            label_time = sample[:,self.colnames.index('cumul_time_s')][-1]
            preds.append(pred_time)
            labels.append(label_time)
        res = {'preds': np.array(preds), 'labels': np.array(labels)}
        return res