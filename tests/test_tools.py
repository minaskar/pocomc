import unittest

import numpy as np

from pocomc.tools import compute_ess


class ESSTestCase(unittest.TestCase):
    def test_ess_single_particle(self):
        self.assertEqual(compute_ess(np.array([1.0])), 1.0)
        self.assertEqual(compute_ess(np.array([251.0])), 1.0)
        self.assertEqual(compute_ess(np.array([-421.0])), 1.0)
        self.assertEqual(compute_ess(np.array([-421.125251])), 1.0)
        self.assertEqual(compute_ess(np.array([0.0])), 1.0)


if __name__ == '__main__':
    unittest.main()
