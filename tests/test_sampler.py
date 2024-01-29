import unittest
import numpy as np

from scipy.stats import norm

from pocomc.sampler import Sampler
from pocomc.prior import Prior

class SamplerTestCase(unittest.TestCase):
    @staticmethod
    def log_likelihood_single(x):
        return np.sum(-0.5 * np.log(2 * np.pi) - 0.5 * x ** 2)

    @staticmethod
    def log_likelihood_vectorized(x):
        # Gaussian log likelihood with mu = 0, sigma = 1
        return np.sum(-0.5 * np.log(2 * np.pi) - 0.5 * x ** 2, axis=1)

    def test_run(self):
        
        n_dim = 2
        prior = Prior(n_dim*[norm(0, 1)])

        sampler = Sampler(
            prior=prior,
            likelihood=self.log_likelihood_single,
            train_config={'epochs': 1},
            random_state=0,
        )
        sampler.run()

    def test_run2(self):
        
        n_dim = 2
        prior = Prior(n_dim*[norm(0, 1)])

        sampler = Sampler(
            prior=prior,
            likelihood=self.log_likelihood_vectorized,
            vectorize=True,
            train_config={'epochs': 1},
            random_state=0,
        )
        sampler.run()


if __name__ == '__main__':
    unittest.main()
