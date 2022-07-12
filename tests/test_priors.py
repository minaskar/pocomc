import unittest
import numpy as np
from pocomc.priors import Uniform


class PlottingTestCase(unittest.TestCase):
    def test_standard_uniform(self):
        np.random.seed(0)

        lower = 0
        upper = 1
        n_dim = 5
        prior = Uniform(lower, upper, n_dim)

        n_samples = 100

        samples = prior.sample(n_samples)
        self.assertEqual(samples.shape, (n_samples, n_dim))
        self.assertTrue(np.all((samples >= lower) & (samples <= upper)))

        log_prob = prior.log_prob(samples)
        self.assertEqual(log_prob.shape, (n_samples,))

        prob = np.exp(log_prob)
        self.assertEqual(prob.shape, (n_samples,))
        self.assertTrue(np.allclose(prob, 1.0))

        n_samples = 1

        samples = prior.sample(n_samples)
        self.assertEqual(samples.shape, (n_samples, n_dim))
        self.assertTrue(np.all((samples >= lower) & (samples <= upper)))

        log_prob = prior.log_prob(samples)
        self.assertEqual(log_prob.shape, (n_samples,))

        prob = np.exp(log_prob)
        self.assertEqual(prob.shape, (n_samples,))
        self.assertTrue(np.allclose(prob, 1.0))

        # Out of bounds sample
        log_prob = prior.log_prob(np.array([1e3, 1e4, 1, 2, 3]))
        self.assertEqual(log_prob, -np.inf)

    def test_general_uniform(self):
        np.random.seed(0)

        lower = np.array([-1, -2, -3])
        upper = np.array([1, 3, 5])
        n_dim = 3
        prior = Uniform(lower, upper)
        gt_prior = 1 / 2 * 1 / 5 * 1 / 8

        n_samples = 100

        samples = prior.sample(n_samples)
        self.assertEqual(samples.shape, (n_samples, n_dim))
        self.assertTrue(np.all((samples >= lower) & (samples <= upper)))

        log_prob = prior.log_prob(samples)
        self.assertEqual(log_prob.shape, (n_samples,))

        prob = np.exp(log_prob)
        self.assertEqual(prob.shape, (n_samples,))
        self.assertTrue(np.allclose(prob, gt_prior))

        n_samples = 1

        samples = prior.sample(n_samples)
        self.assertEqual(samples.shape, (n_samples, n_dim))
        self.assertTrue(np.all((samples >= lower) & (samples <= upper)))

        log_prob = prior.log_prob(samples)
        self.assertEqual(log_prob.shape, (n_samples,))

        prob = np.exp(log_prob)
        self.assertEqual(prob.shape, (n_samples,))
        self.assertTrue(np.allclose(prob, gt_prior))


if __name__ == '__main__':
    unittest.main()
