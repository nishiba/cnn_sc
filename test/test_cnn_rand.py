# coding: utf-8
import unittest
import numpy as np
from cnn_sc.model.cnn_rand import CNNRand


class TestCNNRand(unittest.TestCase):
    def test_run_without_error(self):
        model = CNNRand()
        x_batch = np.array([[1, 2, 3, 0, 0, 0], [4, 5, 6, 0, 0, 0]])
        t_batch = np.array([0, 1])
        loss = model(x_batch, t_batch)


if __name__ == '__main__':
    unittest.main()
