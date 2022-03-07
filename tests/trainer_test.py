import crypto_backend.trainer as Trainer
import unittest
import pandas as pd

class TrainerTester(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.trainer = Trainer('BTC')


if __name__ == '__main__':
    pass
