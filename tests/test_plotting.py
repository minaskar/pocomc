import unittest
import numpy as np
import torch

import pocomc.plotting
from pocomc import Sampler


class PlottingTestCase(unittest.TestCase):
    @staticmethod
    def make_data():
        # Make a dataset to use in tests
        np.random.seed(0)
        n_data = 500
        n_dim = 4
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

    def create_and_run_sampler(self, epochs: int):
        np.random.seed(0)
        torch.manual_seed(0)

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
            train_config={'epochs': epochs}
        )
        sampler.run(prior_samples=x)
        return sampler

    def test_corner(self):
        # Check that making the corner plot does not raise any errors
        sampler = self.create_and_run_sampler(0)
        pocomc.plotting.corner(sampler.results)

        sampler = self.create_and_run_sampler(1)
        pocomc.plotting.corner(sampler.results)

        sampler = self.create_and_run_sampler(2)
        pocomc.plotting.corner(sampler.results)

    def test_trace(self):
        # Check that making the trace plot does not raise any errors
        sampler = self.create_and_run_sampler(0)
        pocomc.plotting.trace(sampler.results)

        sampler = self.create_and_run_sampler(1)
        pocomc.plotting.trace(sampler.results)

        sampler = self.create_and_run_sampler(2)
        pocomc.plotting.trace(sampler.results)

    def test_run(self):
        # Check that making the run plot does not raise any errors
        sampler = self.create_and_run_sampler(0)
        pocomc.plotting.run(sampler.results)

        sampler = self.create_and_run_sampler(1)
        pocomc.plotting.run(sampler.results)

        sampler = self.create_and_run_sampler(2)
        pocomc.plotting.run(sampler.results)


if __name__ == '__main__':
    unittest.main()
