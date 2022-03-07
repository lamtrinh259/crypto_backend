import unittest
import crypto_backend.utils as utils
from crypto_backend.data import get_data
import pandas as pd


class UtilsTester(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        data = pd.DataFrame({'test':[123,124,1256,1226,123]})
        cls.data = data.astype('int64')

    def test_reduceMem(cls):
        data = utils.reduce_mem_usage(cls.data)
        cls.assertEqual(data.test.dtype, 'int16')
