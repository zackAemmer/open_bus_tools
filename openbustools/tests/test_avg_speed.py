import numpy as np
import pandas as pd
import unittest
from openbustools.traveltime import data_loader
from openbustools.traveltime.models.avg_speed import AvgSpeedModel

class TestAvgSpeedModel(unittest.TestCase):

    def setUp(self):
        # Create a sample dataset
        feats_n = np.ones((4, len(data_loader.NUM_FEAT_COLS)))
        feats_n[:,data_loader.NUM_FEAT_COLS.index('calc_speed_m_s')] = [10, 20, 30, 40]
        feats_n[:,data_loader.NUM_FEAT_COLS.index('t_hour')] = [0, 1, 2, 3]
        feats_n[:,data_loader.NUM_FEAT_COLS.index('t_min_of_day')] = [0, 60, 120, 180]
        data = [
            {'feats_n': feats_n},
            {'feats_n': feats_n},
            {'feats_n': feats_n},
        ]
        dataset = data_loader.H5Dataset(data)
        self.model = AvgSpeedModel("test_model", dataset)

    def test_init(self):
        self.assertEqual(self.model.model_name, "test_model")
        self.assertFalse(self.model.is_nn)
        self.assertFalse(self.model.include_grid)
        self.assertEqual(self.model.colnames, data_loader.NUM_FEAT_COLS)
        self.assertAlmostEqual(self.model.speed_mean, 25.0)
        self.assertEqual(self.model.hour_speed_lookup, {'speeds': {0: 10.0, 1: 20.0, 2: 30.0, 3: 40.0}})
        self.assertEqual(self.model.min_speed_lookup, {'speeds': {0: 10.0, 60: 20.0, 120: 30.0, 180: 40.0}})

    def test_predict_hour(self):
        feats_n = np.ones((4, len(data_loader.NUM_FEAT_COLS)))
        feats_n[:,data_loader.NUM_FEAT_COLS.index('calc_speed_m_s')] = [10, 20, 30, 40]
        feats_n[:,data_loader.NUM_FEAT_COLS.index('t_hour')] = [0, 1, 2, 3]
        feats_n[:,data_loader.NUM_FEAT_COLS.index('t_min_of_day')] = [0, 60, 120, 180]
        data = [
            {'feats_n': feats_n},
            {'feats_n': feats_n},
            {'feats_n': feats_n},
        ]
        dataset = data_loader.H5Dataset(data)
        self.model = AvgSpeedModel("test_model", dataset)

        result = self.model.predict(dataset, 'h')

        self.assertEqual(len(result), 3)
        self.assertAlmostEqual(result[0]['preds'], 0.1)
        self.assertAlmostEqual(result[0]['labels'], 1.0)
        self.assertListEqual(result[0]['preds_raw'].tolist(), [0.1, 0.1, 0.1, 0.1])
        self.assertListEqual(result[0]['labels_raw'].tolist(), [1.0, 1.0, 1.0, 1.0])

    def test_predict_minute(self):
        feats_n = np.ones((4, len(data_loader.NUM_FEAT_COLS)))
        feats_n[:,data_loader.NUM_FEAT_COLS.index('calc_speed_m_s')] = [10, 20, 30, 40]
        feats_n[:,data_loader.NUM_FEAT_COLS.index('t_hour')] = [0, 1, 2, 3]
        feats_n[:,data_loader.NUM_FEAT_COLS.index('t_min_of_day')] = [0, 60, 120, 180]
        data = [
            {'feats_n': feats_n},
            {'feats_n': feats_n},
            {'feats_n': feats_n},
        ]
        dataset = data_loader.H5Dataset(data)
        self.model = AvgSpeedModel("test_model", dataset)

        result = self.model.predict(dataset, 'm')

        self.assertEqual(len(result), 3)
        self.assertAlmostEqual(result[0]['preds'], 0.1)
        self.assertAlmostEqual(result[0]['labels'], 1.0)
        self.assertListEqual(result[0]['preds_raw'].tolist(), [0.1, 0.1, 0.1, 0.1])
        self.assertListEqual(result[0]['labels_raw'].tolist(), [1.0, 1.0, 1.0, 1.0])

if __name__ == '__main__':
    unittest.main()