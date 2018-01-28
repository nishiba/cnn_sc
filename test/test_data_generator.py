# coding: utf-8
import unittest
import numpy as np
import pandas as pd

from preprocessing.data_generator import DataGenerator


class TestDataGenerator(unittest.TestCase):
    def test(self):
        pos = pd.DataFrame({'review': ['a b', 'a b c'], 'label': [0, 1]})
        neg = pd.DataFrame([], columns=['review', 'label'])
        data = DataGenerator(positive_dataset=pos, negative_dataset=neg, test_size=0.0)
        train = data.get_train_dataset()
        self.assertEqual(len(train), 2)
        self.assertEqual(train[0][0].shape, (3, ))
        self.assertEqual(train[1][0].shape, (3, ))
        self.assertTrue(train[0][1] in {0, 1})
        self.assertTrue(train[1][1] in {0, 1})


if __name__ == '__main__':
    unittest.main()
