import unittest
from pathlib import Path
import numpy as np
from pocomc import Sampler
from pocomc import Prior

from scipy.stats import norm


class SamplerStateTestCase(unittest.TestCase):
    @staticmethod
    def log_likelihood_vectorized(x):
        # Gaussian log likelihood with mu = 0, sigma = 1
        return np.sum(-0.5 * np.log(2 * np.pi) - 0.5 * x ** 2, axis=1)

    def test_save(self):
        # Save PMC state.
        prior = Prior([norm(0, 1), norm(0, 1)])
        s = Sampler(prior, self.log_likelihood_vectorized, vectorize=True, train_config=dict(epochs=10), random_state=0)
        path = Path('pmc.state')
        s.save_state(path)
        self.assertTrue(path.exists())
        path.unlink()
        self.assertFalse(path.exists())

    def test_load(self):
        # Load PMC state.
        prior = Prior([norm(0, 1), norm(0, 1)])
        s = Sampler(prior, self.log_likelihood_vectorized, vectorize=True, train_config=dict(epochs=10), random_state=0)
        path = Path('pmc.state')
        s.save_state(path)
        self.assertTrue(path.exists())
        s.load_state(path)
        path.unlink()
        self.assertFalse(path.exists())

    def test_resume(self):
        # Run PMC. Then, pick an intermediate state and resume from that state.
        np.random.seed(0)
        prior = Prior([norm(0, 1), norm(0, 1)])
        s = Sampler(prior, self.log_likelihood_vectorized, vectorize=True, train_config=dict(epochs=10), random_state=0)
        s.run(save_every=1)  # Save every iteration

        # At this point, we would look at the directory and choose the file we want to load. In this example, we select
        # "pmc_1.state". Now we rerun the sampler starting from this path. We will not get the exact same
        # results due to RNG.

        self.assertTrue(Path("states/pmc_1.state").exists())
        self.assertTrue(Path("states/pmc_2.state").exists())
        self.assertTrue(Path("states/pmc_3.state").exists())

        s = Sampler(prior, self.log_likelihood_vectorized, vectorize=True, train_config=dict(epochs=10), random_state=0)
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


if __name__ == '__main__':
    unittest.main()
