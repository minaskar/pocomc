import unittest
import numpy as np

from pocomc.scaler import Reparameterise


class ReparameteriseTestCase(unittest.TestCase):
    @staticmethod
    def make_unconstrained_data():
        # Make a dataset to use in tests
        np.random.seed(0)
        n_data = 100
        n_dim = 10
        x = np.random.randn(n_data, n_dim) * 3 + 1

        lower_bound = np.nan
        upper_bound = np.nan
        return x, lower_bound, upper_bound

    @staticmethod
    def make_lower_bounded_data():
        # Make a dataset to use in tests
        np.random.seed(0)
        n_data = 100
        n_dim = 10
        x = np.random.exponential(scale=1, size=(n_data, n_dim))

        lower_bound = 0
        upper_bound = np.nan
        return x, lower_bound, upper_bound

    @staticmethod
    def make_upper_bounded_data():
        # Make a dataset to use in tests
        np.random.seed(0)
        n_data = 100
        n_dim = 10
        x = -np.random.exponential(scale=1, size=(n_data, n_dim))

        lower_bound = np.nan
        upper_bound = 0
        return x, lower_bound, upper_bound

    @staticmethod
    def make_lower_and_upper_bounded_data():
        # Make a dataset to use in tests
        np.random.seed(0)
        n_data = 100
        n_dim = 10
        x = np.random.uniform(low=0, high=1, size=(n_data, n_dim))

        lower_bound = 0
        upper_bound = 1
        return x, lower_bound, upper_bound

    def test_unconstrained(self):
        # Test that methods work without errors on unconstrained input data
        np.random.seed(0)

        x, lb, ub = self.make_unconstrained_data()
        r = Reparameterise(ndim=x.shape[1], bounds=(lb, ub))
        r.fit(x)

        u = r.forward(x)
        x_r, _ = r.inverse(u)

        self.assertEqual(x.shape, u.shape)
        self.assertEqual(x.dtype, u.dtype)

        self.assertEqual(x_r.shape, u.shape)
        self.assertEqual(x_r.dtype, u.dtype)

        self.assertTrue(np.allclose(x, x_r))

    def test_lower_bounded(self):
        # Test that methods work without errors on unconstrained input data
        np.random.seed(0)

        x, lb, ub = self.make_lower_bounded_data()
        r = Reparameterise(ndim=x.shape[1], bounds=(lb, ub))
        r.fit(x)

        u = r.forward(x)
        x_r, _ = r.inverse(u)

        self.assertEqual(x.shape, u.shape)
        self.assertEqual(x.dtype, u.dtype)

        self.assertEqual(x_r.shape, u.shape)
        self.assertEqual(x_r.dtype, u.dtype)

        self.assertTrue(np.allclose(x, x_r))

    def test_upper_bounded(self):
        # Test that methods work without errors on unconstrained input data
        np.random.seed(0)

        x, lb, ub = self.make_upper_bounded_data()
        r = Reparameterise(ndim=x.shape[1], bounds=(lb, ub))
        r.fit(x)

        u = r.forward(x)
        x_r, _ = r.inverse(u)

        self.assertEqual(x.shape, u.shape)
        self.assertEqual(x.dtype, u.dtype)

        self.assertEqual(x_r.shape, u.shape)
        self.assertEqual(x_r.dtype, u.dtype)

        self.assertTrue(np.allclose(x, x_r))

    def test_lower_and_upper_bounded(self):
        # Test that methods work without errors on unconstrained input data
        np.random.seed(0)

        x, lb, ub = self.make_lower_and_upper_bounded_data()
        r = Reparameterise(ndim=x.shape[1], bounds=(lb, ub))
        r.fit(x)

        u = r.forward(x)
        x_r, _ = r.inverse(u)

        self.assertEqual(x.shape, u.shape)
        self.assertEqual(x.dtype, u.dtype)

        self.assertEqual(x_r.shape, u.shape)
        self.assertEqual(x_r.dtype, u.dtype)

        self.assertTrue(np.allclose(x, x_r))


if __name__ == '__main__':
    unittest.main()
