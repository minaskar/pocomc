import torch
import torch.nn as nn

from torch import Tensor
from torch.distributions import Normal
from typing import *
from zuko.utils import odeint

import numpy as np
import copy
import time 
import zuko
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau

from .tools import torch_double_to_float

class MLP(nn.Sequential):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_features: List[int] = [64, 64],
    ):
        layers = []

        for a, b in zip(
            (in_features, *hidden_features),
            (*hidden_features, out_features),
        ):
            layers.extend([nn.Linear(a, b), nn.ELU()])
            #layers.extend([nn.Linear(a, b), nn.ELU(), nn.Dropout(0.2)])

        super().__init__(*layers[:-1])
        #super().__init__(*layers[:-2])


class CNF(nn.Module):
    def __init__(
        self,
        features: int,
        freqs: int = 3,
        **kwargs,
    ):
        super().__init__()

        self.net = MLP(2 * freqs + features, features, **kwargs)

        self.register_buffer('freqs', torch.arange(1, freqs + 1) * torch.pi)

    def forward(self, t: Tensor, x: Tensor) -> Tensor:
        t = self.freqs * t[..., None]
        t = torch.cat((t.cos(), t.sin()), dim=-1)
        t = t.expand(*x.shape[:-1], -1)

        return self.net(torch.cat((t, x), dim=-1))

    def encode(self, x: Tensor) -> Tensor:
        return odeint(self, x, 0.0, 1.0, phi=self.parameters())

    def decode(self, z: Tensor) -> Tensor:
        return odeint(self, z, 1.0, 0.0, phi=self.parameters())

    def log_prob(self, x: Tensor) -> Tensor:
        I = torch.eye(x.shape[-1]).to(x)
        I = I.expand(x.shape + x.shape[-1:]).movedim(-1, 0)

        def augmented(t: Tensor, x: Tensor, ladj: Tensor) -> Tensor:
            with torch.enable_grad():
                x = x.requires_grad_()
                dx = self(t, x)

            jacobian = torch.autograd.grad(dx, x, I, is_grads_batched=True, create_graph=True)[0]
            trace = torch.einsum('i...i', jacobian)

            return dx, trace * 1e-2

        ladj = torch.zeros_like(x[..., 0])
        z, ladj = odeint(augmented, (x, ladj), 0.0, 1.0, phi=self.parameters())

        return Normal(0.0, z.new_tensor(1.0)).log_prob(z).sum(dim=-1) + ladj * 1e2
    
    def _forward(self, x):
        I = torch.eye(x.shape[-1]).to(x)
        I = I.expand(x.shape + x.shape[-1:]).movedim(-1, 0)

        def augmented(t: Tensor, x: Tensor, ladj: Tensor) -> Tensor:
            with torch.enable_grad():
                x = x.requires_grad_()
                dx = self(t, x)

            jacobian = torch.autograd.grad(dx, x, I, is_grads_batched=True, create_graph=True)[0]
            trace = torch.einsum('i...i', jacobian)

            return dx, trace * 1e-2

        ladj = torch.zeros_like(x[..., 0])
        z, ladj = odeint(augmented, (x, ladj), 0.0, 1.0, phi=self.parameters())

        return z, ladj
    
    def _inverse(self, z):
        x = self.decode(z)
        _, ladj = self._forward(x)
        return x, -ladj


class FlowMatchingLoss(nn.Module):
    def __init__(self, v: nn.Module):
        super().__init__()

        self.v = v

    def forward(self, x: Tensor) -> Tensor:
        t = torch.rand_like(x[..., 0]).unsqueeze(-1)
        z = torch.randn_like(x)
        y = (1 - t) * x + (1e-4 + (1 - 1e-4) * t) * z
        u = (1 - 1e-4) * z - x

        return (self.v(t.squeeze(-1), y) - u).square().mean()


class Flow:

    def __init__(self, n_dim, flow=None):
        self.n_dim = n_dim
        self.flow = CNF(n_dim, hidden_features=[256] * 3)
        self.loss = FlowMatchingLoss(self.flow)

    def forward(self, x):
        return self.flow._forward(x)

    def inverse(self, u):
        return self.flow._inverse(u)

    def log_prob(self, x):
        return self.flow.log_prob(x)

    def sample(self, size=1):
        u = torch.randn(size, self.n_dim)
        x, _ = self.flow.decode(u)
        return x, self.log_prob(x)
    
    def fit(self,
            x,
            weights=None,
            validation_split=0.0,
            epochs=1000,
            batch_size=1000,
            patience=20,
            learning_rate=1e-3,
            weight_decay=0,
            laplace_scale=None,
            gaussian_scale=None,
            annealing=True,
            noise=None,
            shuffle=True,
            clip_grad_norm=1.0,
            verbose=0,
            ):
        """

        Parameters
        ----------
        x : ``torch.Tensor``
            Input samples.
        weights : ``torch.Tensor``, optional
            Weights for each sample. Default: ``None``.
        validation_split : ``float``, optional
            Fraction of samples to use for validation. Default: 0.0.
        epochs : ``int``, optional
            Number of epochs. Default: 1000.
        batch_size : ``int``, optional
            Batch size. Default: 1000.
        patience : ``int``, optional
            Number of epochs without improvement before early stopping. Default: 20.
        learning_rate : ``float``, optional
            Learning rate. Default: 1e-3.
        weight_decay : ``float``, optional
            Weight decay. Default: 0.
        laplace_scale : ``float``, optional
            Laplace regularization scale. Default: ``None``.
        gaussian_scale : ``float``, optional
            Gaussian regularization scale. Default: ``None``.
        annealing : ``bool``, optional
            Whether to use learning rate annealing. Default: ``True``.
        noise : ``float``, optional
            Noise scale. Default: ``None``.
        shuffle : ``bool``, optional
            Whether to shuffle samples. Default: ``True``.
        clip_grad_norm : ``float``, optional
            Maximum gradient norm. Default: 1.0.
        verbose : ``int``, optional
            Verbosity level. Default: 0.

        Returns
        -------
        history : ``dict``
            Dictionary with loss history.

        Examples
        --------
        >>> import torch
        >>> import pocomc
        >>> flow = pocomc.Flow(2)
        >>> x = torch.randn(100, 2)
        >>> history = flow.fit(x)    
        """
        x = torch_double_to_float(x)

        n_samples, n_dim = x.shape

        # Check if weights is not None and resample samples according to weights
        if weights is not None:
            weights = torch_double_to_float(weights)
            weights = weights / torch.sum(weights)
            rand_indx = torch.multinomial(weights, n_samples, replacement=True)
            x = x[rand_indx]
            weights = None

        if shuffle:
            rand_indx = torch.randperm(n_samples)
            x = x[rand_indx]
            if weights is not None:
                weights = weights[rand_indx]

        if noise is not None:
            min_dists = torch.empty(n_samples)
            for i in range(n_samples):
                min_dist = torch.linalg.norm(x[i] - x, axis=1)
                min_dists[i] = torch.min(min_dist[min_dist > 0.0])
            mean_min_dist = torch.mean(min_dist)

        if validation_split > 0.0:
            x_train = x[:int(validation_split * n_samples)]
            x_valid = x[int(validation_split * n_samples):]
            if weights is None:
                train_dl = DataLoader(TensorDataset(x_train), batch_size, shuffle)
                val_dl = DataLoader(TensorDataset(x_valid), batch_size, shuffle)
            else:
                weights_train = weights[:int(validation_split * n_samples)]
                weights_valid = weights[int(validation_split * n_samples):]
                train_dl = DataLoader(TensorDataset(x_train, weights_train), batch_size, shuffle)
                val_dl = DataLoader(TensorDataset(x_valid, weights_valid), batch_size, shuffle)
            validation = True
        else:
            x_train = x
            if weights is None:
                train_dl = DataLoader(TensorDataset(x_train), batch_size, shuffle)
            else:
                weights_train = weights
                train_dl = DataLoader(TensorDataset(x_train, weights_train), batch_size, shuffle)
            validation = False

        optimizer = torch.optim.AdamW(self.flow.parameters(), 
                                      learning_rate,
                                      weight_decay=weight_decay,
                                      )

        if annealing:
            scheduler = ReduceLROnPlateau(optimizer, 
                                          mode='min',
                                          factor=0.2,
                                          patience=patience, 
                                          threshold=0.0001, 
                                          threshold_mode='abs', 
                                          min_lr=1e-6,
                                         )

        history = dict()  # Collects per-epoch loss
        history['loss'] = []
        history['val_loss'] = []
        if validation:
            monitor = 'val_loss'
        else:
            monitor = 'loss'

        best_epoch = 0
        best_loss = np.inf
        best_model = copy.deepcopy(self.flow.state_dict())

        start_time_sec = time.time()

        for epoch in range(epochs):
            self.flow.train()
            train_loss = 0.0

            for batch in train_dl:

                optimizer.zero_grad()
                if noise is None:
                    x_ = batch[0]
                else:
                    x_ = batch[0] + noise * mean_min_dist * torch.randn_like(batch[0])
                if weights is None:
                    loss = self.loss(x_)
                else:
                    loss = self.weighted_loss(x_, batch[1])

                if laplace_scale is not None or gaussian_scale is not None:
                    loss -= regularization_loss(self.flow, laplace_scale, gaussian_scale)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.flow.parameters(), clip_grad_norm)
                optimizer.step()

                train_loss += loss.data.item()

            train_loss = train_loss / len(train_dl.dataset)

            history['loss'].append(train_loss)
            
            if validation:
                self.flow.eval()
                val_loss = 0.0

                for batch in val_dl:

                    if noise is None:
                        x_ = batch[0]
                    else:
                        x_ = batch[0] + noise * mean_min_dist * torch.randn_like(batch[0])
                    if weights is None:
                        loss = self.loss(x_)
                    else:
                        loss = self.weighted_loss(x_, batch[1])
                    
                    if laplace_scale is not None or gaussian_scale is not None:
                        loss -= regularization_loss(self.flow, laplace_scale, gaussian_scale)

                    val_loss += loss.data.item()

                val_loss = val_loss / len(val_dl.dataset)

                history['val_loss'].append(val_loss)
        
            if annealing and validation:
                scheduler.step(val_loss)
            elif annealing:
                scheduler.step(train_loss)

            if verbose > 1:
                try:
                    print('Epoch %3d/%3d, train loss: %5.2f, val loss: %5.2f' % (epoch + 1, epochs, train_loss, val_loss))
                except:  # TODO specify type of exception
                    print('Epoch %3d/%3d, train loss: %5.2f' % (epoch + 1, epochs, train_loss))

            # Monitor loss
            if history[monitor][-1] < best_loss:
                best_loss = history[monitor][-1]
                best_epoch = epoch
                best_model = copy.deepcopy(self.flow.state_dict())

            if epoch - best_epoch >= int(1.5 * patience):
                self.flow.load_state_dict(best_model)
                if verbose > 0:
                    print('Finished early after %3d epochs' % best_epoch)
                    print('Best loss achieved %5.2f' % best_loss)
                break
        
        if verbose > 0:
            end_time_sec = time.time()
            total_time_sec = end_time_sec - start_time_sec
            time_per_epoch_sec = total_time_sec / epochs
            print()
            print('Time total:     %5.2f sec' % total_time_sec)
            print('Time per epoch: %5.2f sec' % time_per_epoch_sec)

        return history
