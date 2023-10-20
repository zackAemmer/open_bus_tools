import numpy as np
from obt import data_loader

from obt import data_utils


class PersistentTimeSeqModel:
    def __init__(self, model_name):
        self.model_name = model_name
        self.requires_grid = False
        self.collate_fn = data_loader.persistent_collate
        self.train_time = 0.0
        self.batch_size = 512
        self.is_nn = False
        return None
    def train(self, dataloader, config):
        return None
    def evaluate(self, dataloader, config):
        seq_lens = []
        labels = []
        for i in dataloader:
            seq_lens.extend(i[0])
            labels.extend(i[1])
        preds = [x*30 for x in seq_lens]
        return np.array(labels), np.array(preds)
    def save_to(self, path):
        data_utils.write_pkl(self, path)
        return None