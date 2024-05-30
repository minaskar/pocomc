import unittest
import torch
from pocomc.flow import Flow

class FlowTestCase(unittest.TestCase):
    @staticmethod
    def make_data():
        # Make a dataset to use in tests
        torch.manual_seed(0)
        n_data = 100
        n_dim = 4
        x = torch.randn(size=(n_data, n_dim)) * 1.5
        return x

    @torch.no_grad()
    def test_forward(self):
        """
        Test that the forward pass works without raising an error
        """
        torch.manual_seed(0)

        data = self.make_data()
        flow = Flow(n_dim=data.shape[1], flow='maf3')
        z, _ = flow.forward(data)

        self.assertFalse(torch.any(torch.isnan(z)))
        self.assertFalse(torch.any(torch.isinf(z)))
        self.assertEqual(data.shape, z.shape)
        self.assertEqual(data.dtype, z.dtype)

    @torch.no_grad()
    def test_inverse(self):
        # Test that the inverse pass works without raising an error
        torch.manual_seed(0)

        z = self.make_data()
        flow = Flow(n_dim=z.shape[1], flow='maf3')
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
        flow = Flow(n_dim=x.shape[1], flow='maf3')

        log_prob = flow.log_prob(x)

        self.assertFalse(torch.any(torch.isnan(log_prob)))
        self.assertFalse(torch.any(torch.isinf(log_prob)))
        self.assertEqual(log_prob.shape, (x.shape[0],))
        self.assertEqual(x.dtype, log_prob.dtype)

    @torch.no_grad()
    def test_sample(self):
        # Test that sample works without raising an error
        torch.manual_seed(0)

        x_tmp = self.make_data()
        flow = Flow(n_dim=x_tmp.shape[1], flow='maf3')
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
        flow = Flow(n_dim=x.shape[1], flow='maf3')
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
        flow = Flow(n_dim=x.shape[1], flow='maf3')
        log_prob = flow.log_prob(x)

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
        flow = Flow(n_dim=x.shape[1], flow='maf3')
        with self.assertWarns(UserWarning):
            log_prob = flow.log_prob(x)

        self.assertFalse(torch.any(torch.isnan(log_prob)))
        self.assertFalse(torch.any(torch.isinf(log_prob)))
        self.assertEqual(log_prob.shape, (x.shape[0],))
        self.assertEqual(log_prob.dtype, torch.float32)

    @torch.no_grad()
    def test_logprob_single_example(self):
        # Test logprob when input is a single data point
        torch.manual_seed(0)

        x = self.make_data()
        x = x[0].reshape(1, -1)
        flow = Flow(n_dim=x.shape[1], flow='maf3')
        log_prob = flow.log_prob(x)

        self.assertFalse(torch.any(torch.isnan(log_prob)))
        self.assertFalse(torch.any(torch.isinf(log_prob)))
        self.assertEqual(log_prob.shape, (x.shape[0],))
        self.assertEqual(x.dtype, log_prob.dtype)

    def test_logprob_backward(self):
        # Test backpropagation on the negative log likelihood
        torch.manual_seed(0)

        x = self.make_data()
        flow = Flow(n_dim=x.shape[1], flow='maf3')
        log_prob = flow.log_prob(x)
        nll = -torch.mean(log_prob)
        nll.backward()

        for param in flow.flow.parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad)
            else:
                self.assertIsNone(param.grad)

    @torch.no_grad()
    def test_logprob_inverse(self):
        # Test that the inverse logprob is the negative of the forward logprob
        torch.manual_seed(0)

        x = self.make_data()

        flow = Flow(n_dim=x.shape[1], flow='maf3')
        z, logprob_forward = flow.forward(x)
        _, logprob_inverse = flow.inverse(z)

        #self.assertTrue(torch.allclose(logprob_forward, -logprob_inverse))
        torch.testing.assert_close(logprob_forward, -logprob_inverse)
        self.assertEqual(logprob_forward.shape, logprob_inverse.shape)
        self.assertEqual(logprob_forward.dtype, logprob_inverse.dtype)

    def test_fit(self):
        # Test that fit works without errors and check some basic functions afterwards
        torch.manual_seed(0)

        x = self.make_data()
        flow = Flow(n_dim=x.shape[1], flow='maf3')
        flow.fit(x, epochs=5)

        z, _ = flow.forward(x)
        log_prob = flow.log_prob(x)
        x_samples, _ = flow.sample(x.shape[0])

        self.assertFalse(torch.any(torch.isnan(log_prob)))
        self.assertFalse(torch.any(torch.isinf(log_prob)))
        self.assertEqual(log_prob.shape, (x.shape[0],))
        self.assertEqual(x.dtype, log_prob.dtype)

        self.assertFalse(torch.any(torch.isnan(z)))
        self.assertFalse(torch.any(torch.isinf(z)))
        self.assertEqual(x.shape, z.shape)
        self.assertEqual(x.dtype, z.dtype)

        self.assertFalse(torch.any(torch.isnan(x_samples)))
        self.assertFalse(torch.any(torch.isinf(x_samples)))
        self.assertEqual(x.shape, x_samples.shape)
        self.assertEqual(x.dtype, x_samples.dtype)

    def test_logj(self):
        torch.manual_seed(0)

        x = self.make_data()
        flow = Flow(n_dim=x.shape[1], flow='maf3')

        z, logj_forward = flow.forward(x)
        _, logj_inverse = flow.inverse(z)

        #assert torch.allclose(logj_forward, -logj_inverse, rtol=1e-4)
        torch.testing.assert_close(logj_forward, -logj_inverse)


if __name__ == '__main__':
    unittest.main()
