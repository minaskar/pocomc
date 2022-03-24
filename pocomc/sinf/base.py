import abc
import imp
import pathlib
from copy import deepcopy
from collections import defaultdict
from typing import Optional, List, Tuple, Dict, Union

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn

from .debug import FlowDebugger

from .SINF import *
from .optimize import *
from .GIS import *



class FlowInterface(abc.ABC):
    def __init__(self, debugger: Optional[FlowDebugger] = None):
        """
        Interface class that lets algorithms like DLA access key flow functionalities and update the flow.
        The intended use is to create a specific implementation.
        """
        self.flow = None
        self.debugger = debugger
        self._auto_step: int = 0  # Training epoch counter

        self._z_samples: Optional[torch.Tensor] = None  # For debugging purposes

    @property
    def debug(self):
        return self.debugger is not None

    def debug_step(self):
        if self.debug:
            self.debugger.step()

    def debug_animate(self):
        if self.debug:
            self.debugger.animate()

    @torch.no_grad()
    def add_debug_data(self, x_train: torch.Tensor, x_val: torch.Tensor = None, scalars: dict = None):
        if not self.debug:
            return

        if scalars is None:
            scalars = dict()

        # Check debugger configurations
        writer_config = self.debugger.file_writer.config
        visualizer_config = self.debugger.visualizer.config

        if writer_config.write_scalars or visualizer_config.plot_scalars:
            self.debugger.add_scalar('logq_train', self.logq(x_train).mean())
            if x_val is not None:
                self.debugger.add_scalar('logq_val', self.logq(x_val).mean())
            for key, value in scalars.items():
                self.debugger.add_scalar(key, value)

        if writer_config.write_samples:
            # TODO create necessary method in file_writer
            pass

        if writer_config.write_training_paths or visualizer_config.plot_training_paths:
            paths = self.forward_paths(x_train)
            self.debugger.add_training_paths(paths)

        if writer_config.write_training_reconstructions or visualizer_config.plot_training_reconstructions:
            z_train = self.forward(x_train)
            x_train_reconstructed = self.inverse(z_train)
            self.debugger.add_training_reconstructions(x_train, x_train_reconstructed)

        if (writer_config.write_validation_paths or visualizer_config.plot_validation_paths) and x_val is not None:
            paths = self.forward_paths(x_val)
            self.debugger.add_validation_paths(paths)

        if (
                writer_config.write_validation_reconstructions or visualizer_config.plot_validation_reconstructions) and x_val is not None:
            z_val = self.forward(x_val)
            x_val_reconstructed = self.inverse(z_val)
            self.debugger.add_validation_reconstructions(x_val, x_val_reconstructed)

        if writer_config.write_generative_paths or visualizer_config.plot_generative_paths:
            if self._z_samples is None:
                n_dim = x_train.shape[1]
                self._z_samples = torch.randn(visualizer_config.n_latent_points, n_dim)
            x_generated = self.inverse_paths(self._z_samples)
            self.debugger.add_generative_paths(x_generated)

    @abc.abstractmethod
    def create_flow(self, *args, **kwargs):
        """
        Create the normalizing flow object.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def train_flow(self, *args, **kwargs):
        """
        Train the normalizing flow.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def sample(self, n_samples: int, **kwargs) -> torch.Tensor:
        """
        Sample points in data space from the normalizing flow.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def logq(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the log-density of the flow at the supplied sample locations.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def forward_with_logj(self, x: torch.Tensor, **kwargs):
        """
        Perform the forward pass, returning latent samples and logj.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def inverse_with_logj(self, z: torch.Tensor, **kwargs):
        """
        Perform the inverse pass, returning data space samples and logj.
        """
        raise NotImplementedError

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Push data space samples to latent space.

        :param x: points in data space with shape (n, d).
        :return: points in latent space with shape (n, d).
        """
        return self.forward_with_logj(x, **kwargs)[0]

    def inverse(self, z: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Push latent space samples to data space.

        :param z: points in latent space with shape (n, d).
        :return: points in data space with shape (n, d).
        """
        return self.inverse_with_logj(z, **kwargs)[0]

    def logj_forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Compute the log of the Jacobian determinant at points x (thereby pushing x forward through the flow).
        """
        return self.forward_with_logj(x, **kwargs)[1]

    def logj_backward(self, z: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Compute the log of the Jacobian determinant at points z (thereby pushing z backward through the flow).
        """
        return self.inverse_with_logj(z, **kwargs)[1]

    def grad_x_logq(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Compute gradient of logq wrt x i.e. d/dx (log q(x)).

        This means computing the density of x by:
        * pushing it into latent space;
        * evaluating the density in the latent space;
        * accounting for the change of volume with the Jacobian determinant.
        After obtaining a scalar value (log density of x), we take its derivative wrt x.

        :param x: points in data space with shape (n, d).
        :return: gradient with shape (n, d).
        """
        raise NotImplementedError

    def grad_z_logp(self, z: torch.Tensor, grad_wrt_x, **kwargs) -> torch.Tensor:
        """
        Compute the gradient of logp wrt z.

        We are computing grad_x(logp)dx/dz, so grad_wrt_x should be grad_x(logp).
        Also: grad_z logp = d/dz logp = d/dx dx/dz logp = d/dx logp * dx/dz = grad_wrt_x * dx/dz.

        :param z: points in latent space with shape (n, d).
        :param grad_wrt_x: vector to be used in the Jacobian-vector product.
        :return: gradient of logq wrt z.
        """
        raise NotImplementedError

    def grad_z_logj(self, z: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Compute the gradient of logj wrt z.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def forward_paths(self, x: torch.Tensor) -> torch.Tensor:
        """
        Push x through the flow in the forward direction and return all intermediate states in a tensor.
        If the flow has L layers and x has shape (n, n_dim), then the output has shape (L, n, n_dim).
        """
        raise NotImplementedError

    @abc.abstractmethod
    def inverse_paths(self, z: torch.Tensor):
        """
        Push x through the flow in the inverse direction and return all intermediate states in a tensor.
        If the flow has L layers and z has shape (n, n_dim), then the output has shape (L, n, n_dim).
        """
        raise NotImplementedError


class TorchFlowInterface(FlowInterface, abc.ABC):
    def __init__(self,
                 debugger: Optional[FlowDebugger] = None,
                 device: torch.device = torch.device('cpu'),
                 optimizer_kwargs: dict = None,
                 n_dim: int = None):
        super().__init__(debugger=debugger)
        self.device = device
        self.optimizer_kwargs = dict() if optimizer_kwargs is None else optimizer_kwargs
        self.n_dim = n_dim

    def sample_with_logq(self, n_samples: int, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample points from the flow and obtain their logq.
        """
        if self.n_dim is None:
            raise ValueError("n_dim must be set, but got None")
        z = torch.randn(n_samples, self.n_dim, device=self.device)
        x, logj_backward = self.inverse_with_logj(z)
        logq = -self.n_dim / 2. * torch.log(torch.tensor(2. * np.pi)) - torch.sum(z ** 2, dim=1) / 2 - logj_backward
        return x, logq

    def grad_x_logq(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Compute gradient of logq wrt x i.e. d/dx (log q(x)).

        This means computing the density of x by:
        * pushing it into latent space;
        * evaluating the density in the latent space;
        * accounting for the change of volume with the Jacobian determinant.
        After obtaining a scalar value (log density of x), we take its derivative wrt x.

        :param x: points in data space with shape (n, d).
        :return: gradient with shape (n, d).
        """

        x_tmp = x.to(self.device)
        x_tmp.requires_grad_(True)
        grad_x = torch.autograd.grad(torch.sum(self.logq(x_tmp)), x_tmp)[0]
        x_tmp.requires_grad_(False)
        return grad_x.detach()

    def grad_z_logp(self, z: torch.Tensor, grad_wrt_x, **kwargs) -> torch.Tensor:
        """
        Compute the gradient of logp wrt z.

        We are computing grad_x(logp)dx/dz, so grad_wrt_x should be grad_x(logp).
        Also: grad_z logp = d/dz logp = d/dx dx/dz logp = d/dx logp * dx/dz = grad_wrt_x * dx/dz.

        :param z: points in latent space with shape (n, d).
        :param grad_wrt_x: vector to be used in the Jacobian-vector product.
        :return: gradient of logq wrt z.
        """
        z_tmp = z.to(self.device)
        grad_wrt_x_tmp = grad_wrt_x.to(self.device)

        z_tmp.requires_grad_(True)
        x = self.inverse(z_tmp)
        grad_z = torch.autograd.grad(x, z_tmp, grad_outputs=grad_wrt_x_tmp.to(self.device))[0]
        z_tmp.detach_()

        return grad_z.detach()

    def grad_z_logj(self, z: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Compute the gradient of logj_forward wrt z.
        """
        z_tmp = z.to(self.device)

        z_tmp.requires_grad_(True)
        logj_backward = self.logj_backward(z_tmp)
        log_jacobian = torch.sum(-logj_backward)
        grad_jacobian = torch.autograd.grad(log_jacobian, z_tmp)[0]
        z_tmp.requires_grad_(False)
        return grad_jacobian.detach()

    @abc.abstractmethod
    def get_optimizers(self):
        """
        Get optimizers for flow training.

        :return: list of torch optimizers, corresponding to different parameter groups of the flow.
        """
        raise NotImplementedError



class SINFInterface(TorchFlowInterface):
    def __init__(self,
                 debugger: Optional[FlowDebugger] = None,
                 device: torch.device = torch.device('cpu'),
                 optimizer_kwargs: dict = None):
        """
        SINF interface.
        TODO add documentation.

        :param device: torch device to use for SINF.
        """
        super().__init__(debugger=debugger, device=device, optimizer_kwargs=optimizer_kwargs)

        self.device = device
        self.flow: Optional[SINF] = None
        self.gis_kwargs = dict(verbose=False)

    def create_flow(self, x: torch.Tensor, weights: torch.Tensor = None, val_frac: float = 0.0, **kwargs):
        """
        Create the SINF model.
        """
        for key in kwargs:
            self.gis_kwargs[key] = kwargs[key]
        if self.device == torch.device('cpu'):
            self.gis_kwargs['nocuda'] = True

        self.n_dim = x.shape[1]

        perm = torch.randperm(len(x))
        xp = x[perm]
        Ntrain = int((1.0 - val_frac) * len(x))
        x_train = xp[:Ntrain]
        if weights is not None:
            wp = weights[perm]
            weight_train = wp[:Ntrain]
        else:
            weight_train = None
        if val_frac > 0.0:
            x_val = xp[Ntrain:]
            if weights is not None:
                weight_val = wp[Ntrain:]
        else:
            x_val = None
            weight_val = None

        # We need to deepcopy the inputs, because SINF modifies training data in place.
        self.flow = GIS(
            data_train=deepcopy(x_train).to(self.device),
            weight_train=deepcopy(weight_train.to(self.device)) if weights is not None else None,
            data_validate=deepcopy(x_val.to(self.device)) if val_frac > 0.0 else None,
            weight_validate=deepcopy(weight_val.to(self.device)) if val_frac > 0.0 and weights is not None else None,
            **self.gis_kwargs
        )

    def train_flow(self, x: torch.Tensor, weights: torch.Tensor = None, val_frac: float = 0.0, **kwargs):
        """
        Retrain GIS.

        :param x: training data with shape (n_train, d).
        :param weights: non-negative weights for samples x.
        :param val_frac: Fraction of samples to use for validation.
        """
        _x = x.to(self.device)
        _weights = weights.to(self.device) if weights is not None else weights
        self.create_flow(_x, _weights, val_frac, **kwargs)

    def sample(self, n_samples: int, **kwargs) -> torch.Tensor:
        """
        Sample points in data space from MAF.
        """
        return self.flow.sample(nsample=n_samples, device=self.device)[0]

    def logq(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Evaluate SINF density at supplied data space locations.
        """
        return self.flow.evaluate_density(x.to(self.device))

    def forward_with_logj(self, x: torch.Tensor, **kwargs):
        z, logj_forward = self.flow.forward(data=x.to(self.device))
        return z, logj_forward

    def inverse_with_logj(self, z: torch.Tensor, **kwargs):
        x, logj_backward = self.flow.inverse(data=z.to(self.device))
        logj_backward = -logj_backward  # Correction due to the implementation of SINF
        return x, logj_backward

    @torch.no_grad()
    def forward_paths(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute forward paths for given data. This means having the original data and all intermediate data points,
        given by forward SINF layer transformations.

        Returns: torch.Tensor with shape (n_layers + 1, x.shape[0], x.shape[1]). The first index corresponds to original
        data points, the last index corresponds to latent representations.
        """
        paths = [torch.clone(x.to(self.device))]
        for layer in self.flow.layer:
            paths.append(layer(paths[-1])[0])
        paths = torch.stack(paths)
        return paths

    @torch.no_grad()
    def inverse_paths(self, z: torch.Tensor):
        """
        Compute inverse paths for given latent points. This means having the latent data and all intermediate data
        points, given by inverse SINF layer transformations.

        Returns: torch.Tensor with shape (n_layers + 1, z.shape[0], z.shape[1]). The first index corresponds to latent
        data points, the last index corresponds to data space representations.
        """
        paths = [torch.clone(z.to(self.device))]
        for layer in self.flow.layer[::-1]:
            paths.append(layer.inverse(paths[-1])[0])
        paths = torch.stack(paths)
        return paths

    def get_optimizers(self):
        return []