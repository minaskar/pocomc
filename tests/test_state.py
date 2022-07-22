import unittest
from pathlib import Path
import numpy as np
from pocomc import Sampler


class SamplerStateTestCase(unittest.TestCase):
    @staticmethod
    def log_likelihood_vectorized(x):
        # Gaussian log likelihood with mu = 0, sigma = 1
        return np.sum(-0.5 * np.log(2 * np.pi) - 0.5 * x ** 2, axis=1)

    @staticmethod
    def log_prior_vectorized(x, lower: float = -5.0, upper: float = 5.0):
        # Uniform log prior with bounds (-5, 5)
        n_particles, n_dim = x.shape
        if np.any((x < lower) | (x > upper)):  # If any dimension is out of bounds, the log prior is -infinity
            return np.zeros(n_particles) - np.inf  # FIXME assign -inf only to invalid particles, not all
        else:
            return np.zeros(n_particles) - n_dim * (np.log(upper - lower))

    def test_save(self):
        # Save PMC state.
        s = Sampler(100, 2, lambda x: 0, lambda x: 0)
        path = Path('pmc.state')
        s.save_state(path)
        self.assertTrue(path.exists())
        path.unlink()
        self.assertFalse(path.exists())

    def test_load(self):
        # Load PMC state.
        s = Sampler(100, 2, lambda x: 0, lambda x: 0)
        path = Path('pmc.state')
        s.save_state(path)
        self.assertTrue(path.exists())
        s.load_state(path)
        path.unlink()
        self.assertFalse(path.exists())

    def test_resume(self):
        # Run PMC. Then, pick an intermediate state and resume from that state.
        np.random.seed(0)
        prior_samples = np.random.randn(100, 2)
        s = Sampler(100, 2, self.log_likelihood_vectorized, self.log_prior_vectorized, random_state=0)
        s.run(prior_samples, save_every=1)  # Save every iteration

        # At this point, we would look at the directory and choose the file we want to load. In this example, we select
        # "pmc_1.state". Now we rerun the sampler starting from this path. We will not get the exact same
        # results due to RNG.

        self.assertTrue(Path("states/pmc_1.state").exists())
        self.assertTrue(Path("states/pmc_2.state").exists())
        self.assertTrue(Path("states/pmc_3.state").exists())

        s = Sampler(100, 2, self.log_likelihood_vectorized, self.log_prior_vectorized, random_state=0)
        s.run(resume_state_path="states/pmc_1.state")

        # Remove the generated state files
        #Path("states/pmc_1.state").unlink()
        #Path("states/pmc_2.state").unlink()
        #Path("states/pmc_3.state").unlink()
        p = Path("states").glob('**/*')
        files = [x for x in p if x.is_file()]
        for f in files:
            f.unlink()
        Path("states").rmdir()

    def test_alter_variable(self):
        # Recover old gamma after saving the state to a file
        s = Sampler(100, 2, lambda x: 0, lambda x: 0)
        s.gamma = 0.1
        path = Path('pmc.state')
        s.save_state(path)
        self.assertTrue(path.exists())

        s.gamma = 0.2
        s.load_state(path)
        self.assertEqual(s.gamma, 0.1)

        path.unlink()
        self.assertFalse(path.exists())


if __name__ == '__main__':
    unittest.main()
