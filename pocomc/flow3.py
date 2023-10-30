import math
import torch
import torch.nn as nn

import time
import numpy as np
import copy

from torch import Tensor
from tqdm import tqdm
from typing import *

from torch.distributions import Normal
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau

from .tools import torch_double_to_float

def log_normal(x: Tensor) -> Tensor:
    return -(x.square() + math.log(2 * math.pi)).sum(dim=-1) / 2

import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super(ResidualBlock, self).__init__()
        self.linear1 = nn.Linear(in_features, out_features)
        self.linear2 = nn.Linear(out_features, out_features)
        self.elu = nn.ELU()
        
    def forward(self, x):
        residual = x
        out = self.linear1(x)
        out = self.elu(out)
        out = self.linear2(out)
        out += residual
        out = self.elu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, in_features, out_features, hidden_features):
        super(ResNet, self).__init__()
        self.linear1 = nn.Linear(in_features, hidden_features)
        self.linear2 = nn.Linear(hidden_features, out_features)
        self.residual_blocks = nn.Sequential(
            ResidualBlock(hidden_features, hidden_features),
            ResidualBlock(hidden_features, hidden_features),
            ResidualBlock(hidden_features, hidden_features),
            ResidualBlock(hidden_features, hidden_features),
        )
        self.elu = nn.ELU()
        
    def forward(self, x):
        out = self.linear1(x)
        out = self.elu(out)
        out = self.residual_blocks(out)
        #out = self.elu(out)
        out = self.linear2(out)
        return out

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

        super().__init__(*layers[:-1])

class FFF(nn.Module):
    def __init__(self, features: int, **kwargs):
        super().__init__()

        #self.f = MLP(features, features, **kwargs)  # encoder
        #self.g = MLP(features, features, **kwargs)  # decoder
        self.f = ResNet(features, features, **kwargs)  # encoder
        self.g = ResNet(features, features, **kwargs)  # decoder

    def forward(self, x: Tensor) -> Tensor:
        return self.f(x)

    def sample(self, z: Tensor) -> Tensor:
        return self.g(z)

    def log_prob(self, x: Tensor) -> Tensor:
        I = torch.eye(x.shape[-1], dtype=x.dtype, device=x.device)
        I = I.expand(*x.shape, -1).movedim(-1, 0)

        with torch.enable_grad():
            x = x.clone().requires_grad_()
            z = self.f(x)

        jacobian = torch.autograd.grad(z, x, I, is_grads_batched=True)[0].movedim(0, -1)
        ladj = torch.linalg.slogdet(jacobian).logabsdet

        return log_normal(z) + ladj
    

class FFFLoss(nn.Module):
    def __init__(self, f: nn.Module, g: nn.Module):
        super().__init__()

        self.f = f
        self.g = g

    def forward(self, x: Tensor, beta: float = 10.0, hutchinson: int = 1) -> Tensor:
        x = x.clone().requires_grad_()
        z = self.f(x)
        y = self.g(z)

        l_re = (y - x).square().sum(dim=-1).mean()

        if hutchinson > 1:
            v0 = torch.randn_like(x.expand(hutchinson, *x.shape))
            v1 = torch.autograd.grad(y, z, v0, retain_graph=True, is_grads_batched=True)[0]
            v2 = torch.autograd.grad(z, x, v1, create_graph=True, is_grads_batched=True)[0]
        else:
            v0 = torch.randn_like(x)
            v1 = torch.autograd.grad(y, z, v0, retain_graph=True)[0]
            v2 = torch.autograd.grad(z, x, v1, create_graph=True)[0]

        l_ml = -(v0 * v2).sum(dim=-1).mean() - log_normal(z).mean()

        z_ = torch.randn_like(x)
        l_re_ = (z_ - self.f(self.g(z_))).square().sum(dim=-1).mean()

        return l_ml + beta * (l_re + l_re_)
    
class Flow:

    def __init__(self, n_dim, flow=None):
        self.n_dim = n_dim
        #self.flow = FFF(n_dim, hidden_features=[256] * 3)
        self.flow = FFF(n_dim, hidden_features=256)
        self.loss = FFFLoss(self.flow.f, self.flow.g)

    def forward(self, x):
        z = self.flow.forward(x)
        return z, self.flow.log_prob(x) - log_normal(z)

    def inverse(self, u):
        x = self.flow.sample(u)
        return x, -(self.flow.log_prob(x) - log_normal(u))

    def log_prob(self, x):
        return self.flow.log_prob(x)

    def sample(self, size=1):
        u = torch.randn(size, self.n_dim)
        x = self.flow.sample(u)
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
            rand_indx = torch.multinomial(weights, 10*n_samples, replacement=True)
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
