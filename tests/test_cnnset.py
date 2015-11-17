from __future__ import absolute_import, division, print_function
import unittest
import random

from fastlearners import cNNSet

random.seed(0)

class TestcNNSet(unittest.TestCase):

    def test_nn_xy(self):
        for brute in [True, False]:
            nnset = cNNSet(2, 1, brute=brute)
            for x_i in range(10):
                nnset.add_xy([1.0*x_i, 1.0*x_i], [2.0*x_i])

            for x_i in range(10):
                dists, idxs = nnset.nn_y([2*x_i], k=1)
                nn_y = nnset.get_y(idxs[0])
                self.assertEqual(list(nn_y), [2.0*x_i])

            for x_i in range(10):
                dists, idxs = nnset.nn_x([x_i, x_i], k=1)
                nn_y = nnset.get_y(idxs[0])
                self.assertEqual(list(nn_y), [2.0*x_i])

if __name__ == '__main__':
    unittest.main()
