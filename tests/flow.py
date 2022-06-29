import unittest
import torch

from pocomc import Flow


class FlowTestCase(unittest.TestCase):
    @staticmethod
    def make_data():
        # Make a dataset to use in tests
        torch.manual_seed(0)
        n_data = 100
        n_dim = 10
        x = torch.randn(size=(n_data, n_dim)) * 3 + 1
        return x

    @torch.no_grad()
    def test_forward(self):
        # Test that the forward pass works without raising an error
        torch.manual_seed(0)

        x = self.make_data()
        flow = Flow(ndim=x.shape[1])
        z, _ = flow.forward(x)

        self.assertFalse(torch.any(torch.isnan(z)))
        self.assertFalse(torch.any(torch.isinf(z)))
        self.assertEqual(x.shape, z.shape)
        self.assertEqual(x.dtype, z.dtype)

    @torch.no_grad()
    def test_inverse(self):
        # Test that the inverse pass works without raising an error
        torch.manual_seed(0)

        z = self.make_data()
        flow = Flow(ndim=z.shape[1])
        x, _ = flow.inverse(z)

        self.assertFalse(torch.any(torch.isnan(x)))
        self.assertFalse(torch.any(torch.isinf(x)))
        self.assertEqual(x.shape, z.shape)
        self.assertEqual(x.dtype, z.dtype)

    @torch.no_grad()
    def test_logprob(self):
        # Test that logprob works without raising an error
        torch.manual_seed(0)

        x = self.make_data()
        flow = Flow(ndim=x.shape[1])
        log_prob = flow.logprob(x)

        self.assertFalse(torch.any(torch.isnan(log_prob)))
        self.assertFalse(torch.any(torch.isinf(log_prob)))
        self.assertEqual(log_prob.shape, (x.shape[0],))
        self.assertEqual(x.dtype, log_prob.dtype)

    @torch.no_grad()
    def test_sample(self):
        # Test that sample works without raising an error
        torch.manual_seed(0)

        x_tmp = self.make_data()
        flow = Flow(ndim=x_tmp.shape[1])
        x, _ = flow.sample(x_tmp.shape[0])

        self.assertFalse(torch.any(torch.isnan(x)))
        self.assertFalse(torch.any(torch.isinf(x)))
        self.assertEqual(x.shape, x_tmp.shape)
        self.assertEqual(x.dtype, x_tmp.dtype)

    @torch.no_grad()
    def test_reconstruction(self):
        # Test that latent points are reconstructed to be close enough to data points
        torch.manual_seed(0)

        x = self.make_data()
        flow = Flow(ndim=x.shape[1])
        z, _ = flow.forward(x)
        x_reconstructed, _ = flow.inverse(z)

        self.assertFalse(torch.any(torch.isnan(x_reconstructed)))
        self.assertFalse(torch.any(torch.isinf(x_reconstructed)))
        self.assertEqual(x_reconstructed.shape, x.shape)
        self.assertEqual(x_reconstructed.dtype, x.dtype)
        self.assertTrue(torch.allclose(x, x_reconstructed, atol=1e-5))

    @torch.no_grad()
    def test_logprob_float32(self):
        # Test logprob when input is torch.float
        torch.manual_seed(0)

        x = self.make_data()
        x = x.float()
        flow = Flow(ndim=x.shape[1])
        log_prob = flow.logprob(x)

        self.assertFalse(torch.any(torch.isnan(log_prob)))
        self.assertFalse(torch.any(torch.isinf(log_prob)))
        self.assertEqual(log_prob.shape, (x.shape[0],))
        self.assertEqual(x.dtype, log_prob.dtype)

    @torch.no_grad()
    def test_logprob_float64(self):
        # Test logprob when input is torch.double
        torch.manual_seed(0)

        x = self.make_data()
        x = x.double()
        flow = Flow(ndim=x.shape[1])
        log_prob = flow.logprob(x)

        self.assertFalse(torch.any(torch.isnan(log_prob)))
        self.assertFalse(torch.any(torch.isinf(log_prob)))
        self.assertEqual(log_prob.shape, (x.shape[0],))
        self.assertEqual(x.dtype, log_prob.dtype)

    @torch.no_grad()
    def test_logprob_1d(self):
        # When input is one dimensional, logprob should raise an error
        torch.manual_seed(0)

        x = self.make_data()
        x = x[:, 0].reshape(-1, 1)
        self.assertRaises(ValueError, lambda: Flow(ndim=x.shape[1]))

    @torch.no_grad()
    def test_logprob_single_example(self):
        # Test logprob when input is a single data point
        torch.manual_seed(0)

        x = self.make_data()
        x = x[0].reshape(1, -1)
        flow = Flow(ndim=x.shape[1])
        log_prob = flow.logprob(x)

        self.assertFalse(torch.any(torch.isnan(log_prob)))
        self.assertFalse(torch.any(torch.isinf(log_prob)))
        self.assertEqual(log_prob.shape, (x.shape[0],))
        self.assertEqual(x.dtype, log_prob.dtype)

    def test_logprob_backward(self):
        # Test backpropagation on the negative log likelihood
        torch.manual_seed(0)
        pass


if __name__ == '__main__':
    unittest.main()
