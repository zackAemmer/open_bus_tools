import numpy as np

from openbustools.traveltime import data_loader
from openbustools import data_utils


class TimeTableModel:
    def __init__(self, model_name):
        self.model_name = model_name
        self.is_nn = False
    def evaluate(self, dataloader):
        sch_times = []
        labels = []
        for i in dataloader:
            sch_times.extend(i[0])
            labels.extend(i[1])
        return np.array(labels), np.array(sch_times)