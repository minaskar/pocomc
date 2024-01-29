import unittest

import numpy as np
from scipy.stats import norm

from pocomc.prior import Prior

class PriorTestCase(unittest.TestCase):

    def test_sample(self):
        prior = Prior([norm(0, 1), norm(0, 1)])
        x = prior.rvs(10)
        self.assertEqual(np.shape(x), (10,2))

    def test_log_prob(self):
        prior = Prior([norm(0, 1), norm(0, 1)])
        x = prior.rvs(10)
        log_prob = prior.logpdf(x)
        self.assertIsInstance(log_prob, np.ndarray)

    def test_log_prob2(self):
        prior = Prior([norm(0, 1), norm(0, 1)])
        x = prior.rvs(10)
        log_prob = prior.logpdf(x)
        self.assertEqual(np.shape(log_prob), (10,))

    def test_log_prob3(self):
        prior = Prior([norm(0, 1), norm(0, 1)])
        x = prior.rvs(10)
        log_prob = prior.logpdf(x)
        self.assertTrue(np.all(log_prob < 0))
    
    def test_log_prob4(self):
        prior = Prior([norm(0, 1), norm(0, 1)])
        x = prior.rvs(10)
        log_prob = prior.logpdf(x)
        self.assertTrue(np.all(np.isfinite(log_prob)))

    def test_bounds(self):
        prior = Prior([norm(0, 1), norm(0, 1)])
        bounds = prior.bounds
        self.assertEqual(np.shape(bounds), (2,2))

    def test_bounds2(self):
        prior = Prior([norm(0, 1), norm(0, 1)])
        bounds = prior.bounds
        self.assertTrue(np.all(bounds[:,0] < bounds[:,1]))

    def test_dim(self):
        prior = Prior([norm(0, 1), norm(0, 1)])
        self.assertEqual(prior.dim, 2)


    