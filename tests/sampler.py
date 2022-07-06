import unittest
import numpy as np

from pocomc.sampler import Sampler


class SamplerTestCase(unittest.TestCase):
    @staticmethod
    def make_data():
        # Make a dataset to use in tests
        np.random.seed(0)
        n_data = 100
        n_dim = 10
        x = np.random.uniform(low=-5, high=5, size=(n_data, n_dim))
        return x

    @staticmethod
    def log_likelihood(x):
        # Gaussian log likelihood with mu = 0, sigma = 1
        return np.sum(-0.5 * np.log(2 * np.pi) - 0.5 * x ** 2, axis=1)

    @staticmethod
    def log_prior(x, lower: float = -5.0, upper: float = 5.0):
        # Uniform log prior with bounds (-5, 5)
        n_particles, n_dim = x.shape
        if np.any((x < lower) | (x > upper)):  # If any dimension is out of bounds, the log prior is -infinity
            return np.zeros(n_particles) - np.inf
        else:
            return np.zeros(n_particles) - n_dim * (np.log(upper - lower))

    def test_run(self):
        # Sampler should do a basic run without raising errors
        x = self.make_data()
        n_particles, n_dim = x.shape

        sampler = Sampler(
            nparticles=n_particles,
            ndim=n_dim,
            loglikelihood=self.log_likelihood,
            logprior=self.log_prior,
            vectorize_prior=True,
            vectorize_likelihood=True,
            bounds=np.array([-5.0, 5.0]),
            train_config={'epochs': 1}
        )
        sampler.run(x0=x)


if __name__ == '__main__':
    unittest.main()
