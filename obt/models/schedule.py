import numpy as np

from obt import data_loader
from obt import data_utils


class TimeTableModel:
    def __init__(self, model_name):
        self.model_name = model_name
        self.requires_grid = False
        self.collate_fn = data_loader.schedule_collate
        self.train_time = 0.0
        self.batch_size = 512
        self.is_nn = False
    def train(self, dataloader, config):
        return None
    def evaluate(self, dataloader, config):
        sch_times = []
        labels = []
        for i in dataloader:
            sch_times.extend(i[0])
            labels.extend(i[1])
        return np.array(labels), np.array(sch_times)
    def save_to(self, path):
        data_utils.write_pkl(self, path)
        return None