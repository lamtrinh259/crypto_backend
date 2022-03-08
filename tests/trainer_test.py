from crypto_backend.trainer import Trainer
import unittest
import pandas as pd

class TrainerTester(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.trainer = Trainer('BTC')
        cls.trainer.load_data()

    def test_prophet_predict_type(cls):
        #test to see if returns a dataframe
        pred = cls.trainer.prophecy_predict()
        cls.assertEqual(type(pred['predict']),pd.DataFrame)

    def test_prophet_predict_length(cls):
        #test to see if prediction is 14 days
        pred = cls.trainer.prophecy_predict()
        cls.assertEqual(len(pred['predict']), 14)

    def test_prophet_predict_values(cls):
        #test to see if predictions are non-zero
        pred = cls.trainer.prophecy_predict()
        cls.assertEqual(pred['predict']['yhat'].le(0).sum(),
                        0)

    def test_sarimax_predict_is_Series(cls):
        #tests to check if all results are series
        pred = cls.trainer.sarimax_prediction()
        cls.assertEqual(type(pred['pred']),pd.Series)

    def test_sarimax_predict_upper_is_Series(cls):
        #tests to check if upper is a series
        pred = cls.trainer.sarimax_prediction()
        cls.assertEqual(type(pred['upper']),pd.Series)

    def test_sarimax_predict_is_Series(cls):
        #tests to check if lower is a series
        pred = cls.trainer.sarimax_prediction()
        cls.assertEqual(type(pred['lower']),pd.Series)

    def test_sarimax_predict_length(cls):
        # test to check all are 14 day long
        pred = cls.trainer.sarimax_prediction()
        cls.assertEqual(len(pred['pred']),14)

    def test_sarimax_predict_upper_length(cls):
        # test to check upper is 14 day long
        pred = cls.trainer.sarimax_prediction()
        cls.assertEqual(len(pred['upper']),14)

    def test_sarimax_predict_lower_length(cls):
        # test to check lower is 14 day long
        pred = cls.trainer.sarimax_prediction()
        cls.assertEqual(len(pred['lower']),14)

    def test_sarimax_pred_values_positive(cls):
        # test to check all predictions are nonzero
        pred = cls.trainer.sarimax_prediction()
        cls.assertEqual(pred['pred'].le(0).sum(),  0)

    def test_sarimax_pred_upper_values_positive(cls):
        # test to check upper is nonzero
        pred = cls.trainer.sarimax_prediction()
        cls.assertEqual(pred['upper'].le(0).sum(), 0)

    def test_sarimax_pred_lower_values_positive(cls):
        # test to check lower is nonzero
        pred = cls.trainer.sarimax_prediction()
        cls.assertEqual(pred['lower'].le(0).sum(), 0)

    def test_sarimax_pred_upper_gt_pred(cls):
        # test to check upper>pred
        pred = cls.trainer.sarimax_prediction()
        cls.assertEqual(pred['pred'].gt(pred['upper']).sum(), 0)

    def test_sarimax_pred_pred_gt_lower(cls):
        # test to check pred>lower
        pred = cls.trainer.sarimax_prediction()
        cls.assertEqual(pred['lower'].gt(pred['pred']).sum(), 0)

    def test_LSTM_pred_positive(cls):
        # test to check if the predictions are positive
        pred = cls.trainer.LSTM_predict()
        cls.assertEqual(pred['pred_future_price'].iloc[-14:].gt(0).sum(),
                        14)

    def test_LSTM_pred_past_data_positive(cls):
        # test to check if the predictions are positive
        pred = cls.trainer.LSTM_predict()
        cls.assertEqual(pred['actual_price'].iloc[:-14].lt(0).sum(),
                        0)



if __name__ == '__main__':
    pass
