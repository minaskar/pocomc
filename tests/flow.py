import unittest
import torch


class FlowTestCase(unittest.TestCase):
    @staticmethod
    def make_data():
        # Make a dataset to use in tests
        n_data = 100
        n_dim = 10
        x = torch.randn(size=(n_data, n_dim))
        return x

    @staticmethod
    def make_flow():
        # Make a Flow to use in tests
        n_data = 100
        n_dim = 10
        x = torch.randn(size=(n_data, n_dim))
        return x

    def test_forward(self):
        # Test that the forward pass works without raising an error
        pass

    def test_inverse(self):
        # Test that the inverse pass works without raising an error
        pass

    def test_logprob(self):
        # Test that logprob works without raising an error
        pass

    def test_sample(self):
        # Test that sample works without raising an error
        pass

    def test_reconstruction(self):
        # Test that latent points are reconstructed to be close enough to data points
        pass

    def test_logprob_float32(self):
        # Test logprob when input is torch.float
        pass

    def test_logprob_float64(self):
        # Test logprob when input is torch.double
        pass

    def test_logprob_float(self):
        # Test logprob when input is float (basic python datatype)
        pass


if __name__ == '__main__':
    unittest.main()
