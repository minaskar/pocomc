import math
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D


def create_masks(input_size: int,
                 hidden_size: int,
                 n_hidden: int,
                 input_order: str = 'sequential',
                 input_degrees: torch.Tensor = None):
    """
    Helper function to create masks.
    The masks are used to hide certain connections
    in the neural network thus preserving the 
    autoregressive property.
    
    Parameters
    ----------
    input_size : int
        Dimensionality of input data
    hidden_size : int
        Size of hidden layer
    n_hidden : int
        Number of hidden layers
    input_order : str
        Variable order for creating the autoregressive masks: ``"sequential"`` (default) or ``"random"``.
    input_degrees : torch.Tensor or None
        Degrees of connections between layers
    
    Returns
    -------
    masks : torch.Tensor
        Masks
    degrees : torch.Tensor
        Degrees of connections between layers
    """
    # MADE paper sec 4:
    # degrees of connections between layers -- ensure at most in_degree - 1 connections
    degrees = []

    # set input degrees to what is provided in args (the flipped order of the previous layer in a stack of MADEs);
    # else init input degrees based on strategy in input_order (sequential or random)
    if input_order == 'sequential':
        degrees += [torch.arange(input_size)] if input_degrees is None else [input_degrees]
        for _ in range(n_hidden + 1):
            degrees += [torch.arange(hidden_size) % (input_size - 1)]
        degrees += [torch.arange(input_size) % input_size - 1] if input_degrees is None else [
            input_degrees % input_size - 1]

    elif input_order == 'random':
        degrees += [torch.randperm(input_size)] if input_degrees is None else [input_degrees]
        for _ in range(n_hidden + 1):
            min_prev_degree = min(degrees[-1].min().item(), input_size - 1)
            degrees += [torch.randint(min_prev_degree, input_size, (hidden_size,))]
        min_prev_degree = min(degrees[-1].min().item(), input_size - 1)
        degrees += [torch.randint(min_prev_degree, input_size, (input_size,)) - 1] if input_degrees is None else [
            input_degrees - 1]

    # construct masks
    masks = []
    for (d0, d1) in zip(degrees[:-1], degrees[1:]):
        masks += [(d1.unsqueeze(-1) >= d0.unsqueeze(0)).float()]

    return masks, degrees[0]


class MaskedLinear(nn.Linear):
    """
    Masked Linear Layer (MADE building block layer)

    Parameters
    ----------
    input_size : int
        Dimensionality of input data.
    n_outputs : int
        Number of outputs
    mask : torch.Tensor
        Mask used to hide connections.
    cond_label_size : int
        Number of conditional arguments.
    """
    def __init__(self,
                 input_size: int,
                 n_outputs: int,
                 mask: torch.Tensor,
                 cond_label_size: int = None):
        super().__init__(input_size, n_outputs)

        self.register_buffer('mask', mask)

        self.cond_label_size = cond_label_size
        if cond_label_size is not None:
            self.cond_weight = nn.Parameter(torch.rand(n_outputs, cond_label_size) / math.sqrt(cond_label_size))

    def forward(self,
                x: torch.Tensor,
                y: torch.Tensor = None):
        """
        Forward transformation.

        Parameters
        ----------
        x : torch.Tensor
            Input data.
        y : torch.Tensor
            Conditional input data.
        Returns
        -------
        out : torch.Tensor
            Transformed data.
        """
        out = F.linear(x, self.weight * self.mask, self.bias)
        if y is not None:
            out = out + F.linear(y, self.cond_weight)
        return out

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        ) + (self.cond_label_size is not None) * ', cond_features={}'.format(self.cond_label_size)


class BatchNorm(nn.Module):
    """
    Batch Normalisation layer

    Parameters
    ----------
    input_size : int
        Dimensionality of input data
    momentum : float
        Value of momentum variable. Default: ``0.9``.
    eps : float
        Value of epsilon parameter. Default: ``1e-5``.
    """
    def __init__(self,
                 input_size: int,
                 momentum: float = 0.9,
                 eps: float = 1e-5):
        super().__init__()
        self.momentum = momentum
        self.eps = eps

        self.batch_mean = None
        self.batch_var = None

        self.log_gamma = nn.Parameter(torch.zeros(input_size))
        self.beta = nn.Parameter(torch.zeros(input_size))

        self.register_buffer('running_mean', torch.zeros(input_size))
        self.register_buffer('running_var', torch.ones(input_size))

    def forward(self,
                x: torch.Tensor,
                cond_y: torch.Tensor = None):
        """
        Forward transformation.
        Parameters
        ----------
        x : torch.Tensor
            Input data.
        cond_y : torch.Tensor
            Conditional input data.
        Returns
        -------
        y : torch.Tensor
            Transformed data.
        log_abs_det_jacobian : torch.Tensor
            log(abs(det(Jacobian)))
        """
        if self.training:
            self.batch_mean = x.mean(0)

            # note MAF paper uses biased variance estimate; ie x.var(0, unbiased=False)
            # If x has a single example, we set the variance to 1.
            self.batch_var = x.var(0) if len(x) > 1 else torch.ones_like(self.batch_mean)

            # update running mean
            self.running_mean.mul_(self.momentum).add_(self.batch_mean.data * (1 - self.momentum))
            self.running_var.mul_(self.momentum).add_(self.batch_var.data * (1 - self.momentum))

            mean = self.batch_mean
            var = self.batch_var
        else:
            mean = self.running_mean
            var = self.running_var

        # compute normalized input (cf original batch norm paper algo 1)
        x_hat = (x - mean) / torch.sqrt(var + self.eps)
        y = self.log_gamma.exp() * x_hat + self.beta

        # compute log_abs_det_jacobian (cf RealNVP paper)
        log_abs_det_jacobian = self.log_gamma - 0.5 * torch.log(var + self.eps)
        return y, log_abs_det_jacobian.expand_as(x)

    def inverse(self,
                y: torch.Tensor,
                cond_y: torch.Tensor = None):
        """
        Inverse transformation.
        Parameters
        ----------
        y : torch.Tensor
            Input data.
        cond_y : torch.Tensor
            Conditional input data.
        Returns
        -------
        x : torch.Tensor
            Transformed data.
        log_abs_det_jacobian : torch.Tensor
            log(abs(det(Jacobian)))
        """
        if self.training:
            mean = self.batch_mean if self.batch_mean is not None else torch.tensor(0.0)
            var = self.batch_var if self.batch_var is not None else torch.tensor(1.0)
        else:
            mean = self.running_mean
            var = self.running_var

        x_hat = (y - self.beta) * torch.exp(-self.log_gamma)
        x = x_hat * torch.sqrt(var + self.eps) + mean

        log_abs_det_jacobian = 0.5 * torch.log(var + self.eps) - self.log_gamma

        return x, log_abs_det_jacobian.expand_as(x)


class FlowSequential(nn.Sequential):
    """ Join multiple layers of a normalizing flow """

    def forward(self, x, y):  # TODO handle signature mismatch
        """
        Forward transformation.
        Parameters
        ----------
        x : torch.Tensor
            Input data.
        y : torch.Tensor
            Conditional input data.
        Returns
        -------
        u : torch.Tensor
            Transformed data.
        log_abs_det_jacobian : torch.Tensor
            log(abs(det(Jacobian)))
        """
        sum_log_abs_det_jacobians = 0
        for module in self:
            x, log_abs_det_jacobian = module(x, y)
            sum_log_abs_det_jacobians = sum_log_abs_det_jacobians + log_abs_det_jacobian
        return x, sum_log_abs_det_jacobians

    def inverse(self,
                u: torch.Tensor,
                y: torch.Tensor):
        """
        Inverse transformation.
        Parameters
        ----------
        u : torch.Tensor
            Input data.
        y : torch.Tensor
            Conditional input data.
        Returns
        -------
        x : torch.Tensor
            Transformed data.
        log_abs_det_jacobian : torch.Tensor
            log(abs(det(Jacobian)))
        """
        sum_log_abs_det_jacobians = 0
        for module in reversed(self):
            u, log_abs_det_jacobian = module.inverse(u, y)
            sum_log_abs_det_jacobians = sum_log_abs_det_jacobians + log_abs_det_jacobian
        return u, sum_log_abs_det_jacobians


class MADE(nn.Module):
    r"""Masked Autoregressive Density Estimator (MADE)

    Parameters
    ----------
    input_size : int
        Dimensionality of inputs
    hidden_size : int
        Size of hidden layers
    n_hidden : int
        Number of hidden layers
    activation : str
        Activation function: ``"relu"`` (default) or ``"tanh"``.
    input_order : str
        Variable order for creating the autoregressive masks:
        ``"sequential"`` (default) or ``"random"``.
    """

    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 n_hidden: int,
                 cond_label_size=None,
                 activation: str = 'relu',
                 input_order: str = 'sequential',
                 input_degrees=None):
        super().__init__()
        # base distribution for calculation of log prob under the model
        self.register_buffer('base_dist_mean', torch.zeros(input_size))
        self.register_buffer('base_dist_var', torch.ones(input_size))

        # create masks
        masks, self.input_degrees = create_masks(
            input_size,
            hidden_size,
            n_hidden,
            input_order,
            input_degrees
        )

        # setup activation
        if activation == 'relu':
            activation_fn = nn.ReLU()
        elif activation == 'tanh':
            activation_fn = nn.Tanh()
        else:
            raise ValueError('Check activation function.')

        # construct model
        self.net_input = MaskedLinear(input_size, hidden_size, masks[0], cond_label_size)
        self.net = []
        for m in masks[1:-1]:
            self.net += [activation_fn, MaskedLinear(hidden_size, hidden_size, m)]
        self.net += [activation_fn, MaskedLinear(hidden_size, 2 * input_size, masks[-1].repeat(2, 1))]

        self.net = nn.Sequential(*self.net)

    @property
    def base_dist(self):
        """
        Base distribution
        """
        return D.Normal(self.base_dist_mean, self.base_dist_var)

    def forward(self,
                x: torch.Tensor,
                y: torch.Tensor = None):
        """
        Forward transformation.
        Parameters
        ----------
        x : torch.Tensor
            Input data.
        y : torch.Tensor
            Conditional input data.
        Returns
        -------
        Transformed data.
        """
        # MAF eq 4 -- return mean and log std
        m, loga = self.net(self.net_input(x, y)).chunk(chunks=2, dim=1)
        u = (x - m) * torch.exp(-loga)
        # MAF eq 5
        log_abs_det_jacobian = - loga
        return u, log_abs_det_jacobian

    def inverse(self,
                u: torch.Tensor,
                y: torch.Tensor = None,
                sum_log_abs_det_jacobians=None):
        """Inverse transformation.
        
        Parameters
        ----------
        u : torch.Tensor
            Input data.
        y : torch.Tensor
            Conditional input data.
        sum_log_abs_det_jacobians : torch.Tensor
            Sum of the natural logarithm of the jacobian
            determinant of the transformation.

        Returns
        -------
        Transformed data.
        """
        # MAF eq 3
        x = torch.zeros_like(u)
        # run through reverse model
        loga = 0
        for i in self.input_degrees:
            m, loga = self.net(self.net_input(x, y)).chunk(chunks=2, dim=1)
            x[:, i] = u[:, i] * torch.exp(loga[:, i]) + m[:, i]
        log_abs_det_jacobian = loga
        return x, log_abs_det_jacobian

    def log_prob(self,
                 x: torch.Tensor,
                 y: torch.Tensor = None):
        """
        Log-probability of input data
        Parameters
        ----------
        x : torch.Tensor
            Input data.
        y : torch.Tensor
            Conditional input data.
        Returns
        -------
        Log-probability (log-likelihood) of input.
        """
        u, log_abs_det_jacobian = self.forward(x, y)
        return torch.sum(self.base_dist.log_prob(u) + log_abs_det_jacobian, dim=1)


class MAF(nn.Module):
    """
    Masked Autoregressive Flow (MAF).

    Parameters
    ----------
    n_blocks : int
        Number of MADE blocks.
    input_size : int
        Dimensionality of input data.
    hidden_size : int
        Number of neurons per layer.
    n_hidden : int
        Number of layers per MADE block.
    cond_label_size : int
        Dimensionality of conditional input data.
    activation : str
        Activation function: ``"relu"`` (default) or ``"tanh"``.
    input_order : str
        Variable order for creating the autoregressive masks: ``"sequential"`` (default) or ``"random"``.
    batch_norm : bool
        Whether to use batch normalisation (Default is ``True``).
    """

    def __init__(self,
                 n_blocks: int,
                 input_size: int,
                 hidden_size: int,
                 n_hidden: int,
                 cond_label_size: int = None,
                 activation: str = 'relu',
                 input_order: str = 'sequential',
                 batch_norm: bool = True,
                 **kwargs):
        """
        Parameters
        ----------
        n_blocks : int
            Number of MADE blocks.
        input_size : int
            Input dimensionality.
        hidden_size : int
            Hidden dimension in MADE.
        n_hidden : int
            Number of hidden layers in MADE.
        cond_label_size
        activation : str
            Nonlinearity type.
        input_order
        batch_norm : bool
            Use batch normalization.
        kwargs
        """
        
        # TODO write docstring.
        super().__init__()
        # base distribution for calculation of log prob under the model
        self.register_buffer('base_dist_mean', torch.zeros(input_size))
        self.register_buffer('base_dist_var', torch.ones(input_size))

        # construct model
        modules = []
        self.input_degrees = None
        for i in range(n_blocks):
            modules += [
                MADE(input_size, hidden_size, n_hidden, cond_label_size, activation, input_order, self.input_degrees)]
            self.input_degrees = modules[-1].input_degrees.flip(0)
            modules += batch_norm * [BatchNorm(input_size)]

        self.net = FlowSequential(*modules)

    @property
    def base_dist(self):
        """
        Base distribution
        """
        return D.Normal(self.base_dist_mean, self.base_dist_var, validate_args=False)

    def forward(self,
                x: torch.Tensor,
                y: torch.Tensor = None):
        """
        Forward transformation.
        Parameters
        ----------
        x : torch.Tensor
            Input data.
        y : torch.Tensor
            Conditional input data.
        Returns
        -------
        Transformed data.
        """
        return self.net(x, y)

    def inverse(self,
                u: torch.Tensor,
                y: torch.Tensor = None):
        """
        Inverse transformation.
        Parameters
        ----------
        u : torch.Tensor
            Input data.
        y : torch.Tensor
            Conditional input data.
        Returns
        -------
        Transformed data.
        """
        return self.net.inverse(u, y)

    def log_prob(self,
                 x: torch.Tensor,
                 y: torch.Tensor = None):
        """
        Log-probability of input data
        Parameters
        ----------
        x : torch.Tensor
            Input data.
        y : torch.Tensor
            Conditional input data.
        Returns
        -------
        Log-probability (log-likelihood) of input.
        """
        u, sum_log_abs_det_jacobians = self.forward(x, y)
        return torch.sum(self.base_dist.log_prob(u) + sum_log_abs_det_jacobians, dim=1)

    # TODO only implement a single log_prior for both MAF and RealNVP instead of two separate implementations.
    def log_prior(self, scale: float = 1.0, type: str = 'Laplace'):  # TODO rename `type` to avoid builtin shadow
        """
        Log-prior of weights
        Parameters
        ----------
        scale : float
            Scale parameter for prior.
        type : str
            Type of prior to use. Options are ``"Laplace"`` and ``"Gaussian"``.
        Returns
        -------
        Log-prior of weights.
        """
        total = 0.0

        for i, layer in enumerate(self.net):
            if isinstance(layer, MADE):
                for parameter_name, parameter in layer.net.named_parameters():
                    if parameter_name.endswith('weight'):
                        # Regularize weights, but not biases
                        if type in ['Laplace', 'laplace', 'l1', 'L1']:
                            total += parameter.abs().sum()  # Laplace prior
                        elif type in ['Gaussian', 'gaussian', 'l2', 'L2', 'Normal', 'normal']:
                            total += parameter.square().sum()  # Gaussian prior

        if type in ['Laplace', 'laplace', 'l1', 'L1']:
            return -total / scale
        elif type in ['Gaussian', 'gaussian', 'l2', 'L2', 'Normal', 'normal']:
            return -total / (2 * (scale ** 2))


class LinearMaskedCoupling(nn.Module):
    """
    Modified RealNVP Coupling Layers per the MAF paper

    Parameters
    ----------
    input_size : int
        Dimensionality of input data.
    hidden_size : int
        Number of neurons per layer.
    n_hidden : int
        Number of layers per MADE block.
    mask : torch.Tensor
        Mask used to hide connections.
    cond_label_size : int
        Dimensionality of conditional input data.
    """
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 n_hidden: int,
                 mask: torch.Tensor,
                 cond_label_size: int = None):
        
        super().__init__()

        self.register_buffer('mask', mask)

        # scale function
        s_net = [nn.Linear(input_size + (cond_label_size if cond_label_size is not None else 0), hidden_size)]
        for _ in range(n_hidden):
            s_net += [nn.Tanh(), nn.Linear(hidden_size, hidden_size)]
        s_net += [nn.Tanh(), nn.Linear(hidden_size, input_size)]
        self.s_net = nn.Sequential(*s_net)

        # translation function
        self.t_net = copy.deepcopy(self.s_net)
        # replace Tanh with ReLU's per MAF paper
        for i in range(len(self.t_net)):
            if not isinstance(self.t_net[i], nn.Linear): self.t_net[i] = nn.ReLU()

    def forward(self,
                x: torch.Tensor,
                y: torch.Tensor = None):
        """
        Forward transformation.
        Parameters
        ----------
        x : torch.Tensor
            Input data.
        y : torch.Tensor
            Conditional input data.
        Returns
        -------
        u : torch.Tensor
            Transformed data.
        log_abs_det_jacobian : torch.Tensor
            log(abs(det(Jacobian)))
        """
        # apply mask
        mx = x * self.mask

        # run through model
        s = self.s_net(mx if y is None else torch.cat([y, mx], dim=1))
        t = self.t_net(mx if y is None else torch.cat([y, mx], dim=1))

        # cf RealNVP eq 8 where u corresponds to x (here we're modeling u)
        u = mx + (1 - self.mask) * (x - t) * torch.exp(-s)

        # log det du/dx; cf RealNVP 8 and 6; note, sum over input_size done at model log_prob
        log_abs_det_jacobian = - (1 - self.mask) * s

        return u, log_abs_det_jacobian

    def inverse(self,
                u: torch.Tensor,
                y: torch.Tensor = None):
        """
        Inverse transformation.
        Parameters
        ----------
        u : torch.Tensor
            Input data.
        y : torch.Tensor
            Conditional input data.
        Returns
        -------
        x : torch.Tensor
            Transformed data.
        log_abs_det_jacobian : torch.Tensor
            log(abs(det(Jacobian)))
        """
        # apply mask
        mu = u * self.mask

        # run through model
        s = self.s_net(mu if y is None else torch.cat([y, mu], dim=1))
        t = self.t_net(mu if y is None else torch.cat([y, mu], dim=1))
        x = mu + (1 - self.mask) * (u * s.exp() + t)  # cf RealNVP eq 7

        log_abs_det_jacobian = (1 - self.mask) * s  # log det dx/du

        return x, log_abs_det_jacobian


class RealNVP(nn.Module):
    r"""
    RealNVP normalising flow.

    Parameters
    ----------
    n_blocks : int
        Number of MADE blocks.
    input_size : int
        Dimensionality of input data.
    hidden_size : int
        Number of neurons per layer.
    n_hidden : int
        Number of layers per MADE block.
    cond_label_size : int
        Dimensionality of conditional input data.
    batch_norm : bool
        Whether to use batch normalisation. Default: ``True``.
    """
    def __init__(self,
                 n_blocks: int,
                 input_size: int,
                 hidden_size: int,
                 n_hidden: int,
                 cond_label_size: int = None,
                 batch_norm: bool = True,
                 **kwargs):
        super().__init__()

        # base distribution for calculation of log prob under the model
        self.register_buffer('base_dist_mean', torch.zeros(input_size))
        self.register_buffer('base_dist_var', torch.ones(input_size))

        # construct model
        modules = []
        mask = torch.arange(input_size).float() % 2
        for i in range(n_blocks):
            modules += [LinearMaskedCoupling(input_size, hidden_size, n_hidden, mask, cond_label_size)]
            mask = 1 - mask
            modules += batch_norm * [BatchNorm(input_size)]

        self.net = FlowSequential(*modules)

    @property
    def base_dist(self):
        """
        Base distribution
        """
        return D.Normal(self.base_dist_mean, self.base_dist_var)

    def forward(self,
                x: torch.Tensor,
                y: torch.Tensor = None):
        """
        Forward transformation.
        Parameters
        ----------
        x : torch.Tensor
            Input data.
        y : torch.Tensor
            Conditional input data.
        Returns
        -------
        Transformed data.
        """
        return self.net(x, y)

    def inverse(self,
                u: torch.Tensor,
                y: torch.Tensor = None):
        """
        Inverse transformation.
        Parameters
        ----------
        u : torch.Tensor
            Input data.
        y : torch.Tensor
            Conditional input data.
        Returns
        -------
        Transformed data.
        """
        return self.net.inverse(u, y)

    def log_prob(self,
                 x: torch.Tensor,
                 y: torch.Tensor = None):
        """
        Log-probability of input data
        Parameters
        ----------
        x : torch.Tensor
            Input data.
        y : torch.Tensor
            Conditional input data.
        Returns
        -------
        Log-probability (log-likelihood) of input.
        """
        u, sum_log_abs_det_jacobians = self.forward(x, y)
        return torch.sum(self.base_dist.log_prob(u) + sum_log_abs_det_jacobians, dim=1)

    def log_prior(self,
                  scale: float = 1.0,
                  type: str = 'Laplace'):  # TODO rename `type` to avoid builtin shadow
        """
        Log-prior of weights
        Parameters
        ----------
        scale : float
            Scale parameter for prior.
        type : str
            Type of prior to use. Options are ``"Laplace"`` and ``"Gaussian"``.
        Returns
        -------
        Log-prior of weights.
        """
        total = 0.0
        for i, layer in enumerate(self.net):
            if isinstance(layer, MADE):
                for parameter_name, parameter in layer.net.named_parameters():
                    if parameter_name.endswith('weight'):
                        # Regularize weights, but not biases
                        if type in ['Laplace', 'laplace', 'l1', 'L1']:
                            total += parameter.abs().sum()  # Laplace prior
                        elif type in ['Gaussian', 'gaussian', 'l2', 'L2', 'Normal', 'normal']:
                            total += parameter.square().sum()  # Gaussian prior

        if type in ['Laplace', 'laplace', 'l1', 'L1']:
            return -total / scale
        elif type in ['Gaussian', 'gaussian', 'l2', 'L2', 'Normal', 'normal']:
            return -total / (2 * (scale ** 2))
