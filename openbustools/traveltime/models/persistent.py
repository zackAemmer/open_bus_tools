import numpy as np
from openbustools.traveltime import data_loader

from openbustools import data_utils


class PersistentTimeSeqModel:
    def __init__(self, model_name):
        self.model_name = model_name
        self.is_nn = False
    def evaluate(self, dataloader, config):
        seq_lens = []
        labels = []
        for i in dataloader:
            seq_lens.extend(i[0])
            labels.extend(i[1])
        preds = [x*30 for x in seq_lens]
        return np.array(labels), np.array(preds)