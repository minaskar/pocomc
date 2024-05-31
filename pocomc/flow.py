from typing import Union, Optional, Tuple, Dict, List

import numpy as np
import copy
import time 
import zuko
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau

from .tools import torch_double_to_float

class Flow:
    """
    Normalizing flow model.

    Parameters
    ----------
    n_dim : ``int``
        Number of dimensions of the distribution to be modeled.
    flow : ``zuko.flows.Flow`` or str, optional
        Normalizing flow model. Default: ``nsf3``.

    Attributes
    ----------
    n_dim : ``int``
        Number of dimensions of the distribution to be modeled.
    flow : ``zuko.flows.Flow``
        Normalizing flow model.
    transform : ``zuko.transforms.Transform``
        Transformation object.
    
    Examples
    --------
    >>> import torch
    >>> import pocomc
    >>> flow = pocomc.Flow(2)
    >>> x = torch.randn(100, 2)
    >>> u, logdetj = flow(x)
    >>> x_, logdetj_ = flow.inverse(u)
    >>> log_prob = flow.log_prob(x)
    >>> x_, log_prob_ = flow.sample(100)
    >>> history = flow.fit(x)
    """

    def __init__(self, n_dim, flow='nsf3'):
        self.n_dim = n_dim

        def next_power_of_2(n):
            return 1 if n == 0 else 2**(n - 1).bit_length()
        
        n_hidden = np.maximum(next_power_of_2(3*n_dim), 32)

        if flow == 'maf3':
            self.flow = zuko.flows.MAF(n_dim, 
                                       transforms=3, 
                                       hidden_features=[n_hidden] * 3,
                                       residual=True,)
        elif flow == 'maf6':
            self.flow = zuko.flows.MAF(n_dim, 
                                       transforms=6, 
                                       hidden_features=[n_hidden] * 3,
                                       residual=True,)
        elif flow == 'maf12':
            self.flow = zuko.flows.MAF(n_dim, 
                                       transforms=12, 
                                       hidden_features=[n_hidden] * 3,
                                       residual=True,)
        elif flow == 'nsf3':
            self.flow = zuko.flows.NSF(features=n_dim, 
                                       bins=8, 
                                       transforms=3, 
                                       hidden_features=[n_hidden] * 3,
                                       residual=True)
        elif flow == 'nsf6':
            self.flow = zuko.flows.NSF(features=n_dim, 
                                       bins=8, 
                                       transforms=6, 
                                       hidden_features=[n_hidden] * 3,
                                       residual=True)
        elif flow == 'nsf12':
            self.flow = zuko.flows.NSF(features=n_dim, 
                                       bins=8, 
                                       transforms=12, 
                                       hidden_features=[n_hidden] * 3,
                                       residual=True)
        elif isinstance(flow, zuko.flows.Flow):
            self.flow = flow
        else:
            raise ValueError('Invalid flow type. Choose from: maf3, maf6, maf12, nsf3, nsf6, nsf12, or provide a zuko.flows.Flow object.')

    @property
    def transform(self):
        """
        Transformation object.
        """
        return self.flow().transform

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward transformation.
        Inputs are transformed from the original (relating to the distribution to be modeled) to the latent space.

        Parameters
        ----------
        x : ``torch.Tensor``
            Samples to transform.
        Returns
        -------
        u : ``torch.Tensor``
            Transformed samples in latent space with the same shape as the original space inputs.
        """
        x = torch_double_to_float(x)
        return self.transform.call_and_ladj(x)

    def inverse(self, u: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Inverse transformation.
        Inputs are transformed from the latent to the original space (relating to the distribution to be modeled).
        
        Parameters
        ----------
        u : ``torch.Tensor``
            Samples to transform.
        Returns
        -------
        x : ``torch.Tensor``
            Transformed samples in the original space with the same shape as the latent space inputs.
        """
        u = torch_double_to_float(u)
        x, logdetj = self.transform.inv.call_and_ladj(u)
        return x, logdetj

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute log probability of samples.
        
        Parameters
        ----------
        x : ``torch.Tensor``
            Input samples
        Returns
        -------
        Log-probability of samples.
        """
        x = torch_double_to_float(x)
        return self.flow().log_prob(x)

    def sample(self, size: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Draw random samples from the normalizing flow.

        Parameters
        ----------
        size : ``int``
            Number of samples to generate. Default: 1.
        Returns
        -------
        samples, log_prob : ``tuple``
            Samples as a ``torch.Tensor`` with shape ``(size, n_dimensions)`` and log probability values with shape ``(size, )``.
        """
        x, log_p = self.flow().rsample_and_log_prob((size,))
        return x, log_p

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
                    loss = -self.flow().log_prob(x_).sum()
                else:
                    loss = -self.flow().log_prob(x_) * batch[1] * 1000.0
                    loss = loss.sum() / batch[1].sum()

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
                        loss = -self.flow().log_prob(x_).sum()
                    else:
                        loss = -self.flow().log_prob(x_) * batch[1] * 1000.0
                        loss = loss.sum() / batch[1].sum()
                    
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


def regularization_loss(model, laplace_scale=None, gaussian_scale=None):
    """
    Compute regularization loss.

    Parameters
    ----------
    model : ``zuko.flows.Flow``
        Normalizing flow model.
    laplace_scale : ``float``, optional
        Laplace regularization scale. Default: ``None``.
    gaussian_scale : ``float``, optional
        Gaussian regularization scale. Default: ``None``.
    
    Returns
    -------
    Regularization loss.
    """
    total_laplace = 0.0
    total_gaussian = 0.0

    for i, transform in enumerate(model.transforms):
        if hasattr(transform, "hyper"):
            for parameter_name, parameter in transform.hyper.named_parameters():
                if parameter_name.endswith('weight'):
                    if laplace_scale is not None:
                        total_laplace += parameter.abs().sum()
                    if gaussian_scale is not None:
                        total_gaussian += parameter.square().sum()
    
    total = 0.0
    if laplace_scale is not None:
        total += - total_laplace / laplace_scale
    if gaussian_scale is not None:
        total += - total_gaussian / (2.0 * gaussian_scale**2.0)

    return total