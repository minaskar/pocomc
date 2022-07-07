import unittest
from pocomc.mcmc import Pearson
import numpy as np


class PearsonTestCase(unittest.TestCase):
    def make_data(self):
        # Make a dataset to use in tests
        np.random.seed(0)
        n_data = 100
        n_dim = 5
        x = np.random.randn(n_data, n_dim) * 3 + 1
        return x

    def test_pearson(self):
        np.random.seed(0)
        x = self.make_data()
        y = x + np.random.randn(*x.shape)

        p = Pearson(x)
        output = p.get(y)

        self.assertEqual(output.shape, (5, ))
        for i in range(x.shape[1]):
            self.assertAlmostEqual(output[i], np.corrcoef(x[:, i], y[:, i])[0, 1])


if __name__ == '__main__':
    unittest.main()
