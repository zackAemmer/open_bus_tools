import numpy as np
import pandas as pd
from pathlib import Path
import unittest

from openbustools import standardfeeds
from openbustools.traveltime import data_loader
from openbustools.traveltime.models.avg_speed import AvgSpeedModel


class TestAvgSpeedModel(unittest.TestCase):

    def setUp(self):
        self.days = [x.split(".")[0] for x in standardfeeds.get_date_list("2024_03_15", 3)]
        self.dataset = data_loader.NumpyDataset(
            [Path("openbustools", "tests", "test_data", "kcm_realtime", "processed")],
            self.days,
            load_in_memory=False
        )
        self.model = AvgSpeedModel("test_model", self.dataset)

    def test_init(self):
        self.assertEqual(self.model.model_name, "test_model")
        self.assertIsNone(self.model.config)
        self.assertIsNone(self.model.holdout_routes)
        self.assertFalse(self.model.is_nn)
        self.assertFalse(self.model.include_grid)
        self.assertEqual(self.model.colnames, data_loader.NUM_FEAT_COLS)
        self.assertEqual(len(self.model.hour_speed_lookup), 24)

    def test_predict(self):
        idx = [0, 1, 2]  # Example indices for prediction
        result = self.model.predict(self.dataset, idx)
        self.assertIsInstance(result, dict)
        self.assertIn("preds", result)
        self.assertIn("labels", result)
        self.assertEqual(len(result["preds"]), len(idx))
        self.assertEqual(len(result["labels"]), len(idx))


if __name__ == '__main__':
    unittest.main()