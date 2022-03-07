import unittest
import crypto_backend.transformers as trfmer
# from crypto_backend.data import get_data
import pandas as pd
import numpy as np


class TransformerTester(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = pd.DataFrame({'test':[1, 10, 11, 112, 123],
                                 'test2':[1, 10, 11, 112, 123]})


    def test_log_tr(self):
        #testing to log function
        temp_data = trfmer.LogTransformer().transform(self.data)
        self.assertEqual(temp_data['test'].iloc[0],0)


    def test_log_invtr(cls):
        #testing inverse_log function
        data = trfmer.LogTransformer().inverse_transform(cls.data)
        cls.assertEqual(data.test.iloc[2],np.exp(11))

if __name__ == '__main__':
    tr = TransformerTester()
    tr.test_log_tr()
