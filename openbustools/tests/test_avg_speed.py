import numpy as np
import pandas as pd
import unittest
from openbustools.traveltime import data_loader

class TestAvgSpeedModel(unittest.TestCase):

    def setUp(self):
        # Create a sample dataset
        dataset = data_loader.Dataset()
        dataset.data = [
            {'feats_n': np.array([[10, 0, 0, 0], [20, 1, 1, 1], [30, 2, 2, 2]])},
            {'feats_n': np.array([[40, 3, 3, 3], [50, 4, 4, 4], [60, 5, 5, 5]])},
            {'feats_n': np.array([[70, 6, 6, 6], [80, 7, 7, 7], [90, 8, 8, 8]])}
        ]

        self.model = AvgSpeedModel("test_model", dataset)

    def test_init(self):
        self.assertEqual(self.model.model_name, "test_model")
        self.assertFalse(self.model.is_nn)
        self.assertFalse(self.model.include_grid)
        self.assertEqual(self.model.colnames, data_loader.NUM_FEAT_COLS)
        self.assertAlmostEqual(self.model.speed_mean, 50.0)
        self.assertEqual(self.model.hour_speed_lookup, {0: 15.0, 1: 25.0, 2: 35.0, 3: 45.0, 4: 55.0, 5: 65.0, 6: 75.0, 7: 85.0, 8: 95.0})
        self.assertEqual(self.model.min_speed_lookup, {0: 15.0, 1: 25.0, 2: 35.0, 3: 45.0, 4: 55.0, 5: 65.0, 6: 75.0, 7: 85.0, 8: 95.0})

    def test_predict_hour(self):
        dataset = data_loader.Dataset()
        dataset.data = [
            {'feats_n': np.array([[10, 0, 0, 0], [20, 1, 1, 1], [30, 2, 2, 2]])},
            {'feats_n': np.array([[40, 3, 3, 3], [50, 4, 4, 4], [60, 5, 5, 5]])},
            {'feats_n': np.array([[70, 6, 6, 6], [80, 7, 7, 7], [90, 8, 8, 8]])}
        ]

        result = self.model.predict(dataset, 'h')

        self.assertEqual(len(result), 3)
        self.assertAlmostEqual(result[0]['preds'], [0.66666667, 0.8, 0.85714286])
        self.assertAlmostEqual(result[0]['labels'], 90)
        self.assertAlmostEqual(result[0]['preds_raw'], [0.66666667, 0.8, 0.85714286])
        self.assertAlmostEqual(result[0]['labels_raw'], [10, 20, 30])

    def test_predict_minute(self):
        dataset = data_loader.Dataset()
        dataset.data = [
            {'feats_n': np.array([[10, 0, 0, 0], [20, 1, 1, 1], [30, 2, 2, 2]])},
            {'feats_n': np.array([[40, 3, 3, 3], [50, 4, 4, 4], [60, 5, 5, 5]])},
            {'feats_n': np.array([[70, 6, 6, 6], [80, 7, 7, 7], [90, 8, 8, 8]])}
        ]

        result = self.model.predict(dataset, 'm')

        self.assertEqual(len(result), 3)
        self.assertAlmostEqual(result[0]['preds'], [0.66666667, 0.8, 0.85714286])
        self.assertAlmostEqual(result[0]['labels'], 90)
        self.assertAlmostEqual(result[0]['preds_raw'], [0.66666667, 0.8, 0.85714286])
        self.assertAlmostEqual(result[0]['labels_raw'], [10, 20, 30])

if __name__ == '__main__':
    unittest.main()