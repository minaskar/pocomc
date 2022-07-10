from typing import Union, Optional, Tuple, Dict, List

from .maf import MAF, RealNVP
import torch
from .tools import torch_double_to_float
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import copy
import time


def compute_loss(model: Union[MAF, RealNVP],
                 batch: torch.Tensor,
                 device: str,
                 laplace_prior_scale: Optional[float],
                 gaussian_prior_scale: Optional[float],
                 use_context: bool):
    """
    Compute normalising flow loss given a batch of data.
    The loss is defined as the sum of the negative log likelihood and the negative log prior.
    The likelihood is computed according to the normalising flow.
    The prior is the sum of Laplace and Gaussian priors. The prior is unused by default.

    Parameters
    ----------
    model: ``MAF or RealNVP``
        Normalising flow model used to compute the loss.
    batch: ``torch.Tensor``
        Data with shape (n_samples, n_dimensions).
    device: ``str``
        Torch device used for the loss computation.
    laplace_prior_scale: ``float``
        Laplace prior scale.
    gaussian_prior_scale: ``float``
        Gaussian prior scale.
    use_context: ``bool``
        If True, the batch should be a tuple with data as the first element and context data as the second element.
        The likelihood is then further conditional on context data, not only the parameters of the flow.

    Returns
    -------
    Scalar loss value.
    """
    if use_context:
        x = batch[0].to(device)
        y = batch[1].to(device)
        loss = -model.log_prob(x, y).sum()
    else:
        x = batch[0].to(device)
        loss = -model.log_prob(x).sum()

    if laplace_prior_scale is not None:
        loss -= model.log_prior(scale=laplace_prior_scale, type='Laplace')

    if gaussian_prior_scale is not None:
        loss -= model.log_prior(scale=gaussian_prior_scale, type='Gaussian')

    return loss


def fit(model: Union[MAF, RealNVP],
        data: Union[np.ndarray, torch.Tensor],
        context: Union[np.ndarray, torch.Tensor] = None,
        validation_data: Union[np.ndarray, torch.Tensor] = None,
        validation_context: Union[np.ndarray, torch.Tensor] = None,
        validation_split: float = 0.0,
        epochs: int = 20,
        batch_size: int = 100,
        patience: int = np.inf,
        monitor: str = 'val_loss',
        shuffle: bool = True,
        lr: float = 1e-3,
        weight_decay: float = 1e-8,
        clip_grad_norm: float = 1.0,
        laplace_prior_scale: float = 0.2,
        gaussian_prior_scale: float = None,
        device: str = 'cpu',
        verbose: int = 2):
    r"""
    Fit a normalising flow model.
    We minimize the negative log likelihood of data with an optional L1 and/or L2 regularization term.
    The fitting is done using Adam.

    Parameters
    ----------
    model: ``MAF or RealNVP``
        Normalising flow model to fit.
    data : ``np.ndarray``
        Samples used for training the flow.
    context : ``np.ndarray`` or ``None``
        Additional samples corresponding to conditional  arguments. Default: ``None``.
    validation_data : ``np.ndarray`` or ``None``
        Samples used for validating the flow. Default: ``None``.
    validation_context : ``np.ndarray`` or ``None``
        Additional samples corresponding to conditional arguments, used for validation. Default: ``None``.
    validation_split : ``float``
        Percentage of ``data`` to be used for validation. Default: ``0.0``.
    epochs : ``int``
        Number of training epochs. Default: ``20``.
    batch_size : ``int``
        Batch size used for training. Default: ``100``.
    patience : ``int``
        Number of epochs to wait with no improvement in monitored loss until termination.
        Default: ``np.inf`` (never terminate early).
    monitor : ``str``
        Which loss to monitor for early stopping. Must be one of ``val_loss``, ``loss``. Default: ``val_loss``.
    shuffle : ``bool``
        Shuffle data before training. Default: ``True``.
    lr : ``float``
        Learning rate for Adam. Default: ``1e-3``.
    weight_decay : ``float``
        Weight decay for Adam. Default: ``1e-8``.
    clip_grad_norm : ``float``
        Clip large gradients. Default: ``0``.
    laplace_prior_scale : ``float`` or ``None``
        Laplace prior scale for regularisation. Must be nonnegative. Smaller values correspond to more regularization.
        Default: 0.2.
    gaussian_prior_scale : ``float`` or ``None``
        Gaussian prior scale for regularisation. Must be nonnegative. Smaller values correspond to more regularization.
        Default: None (no regularization).
    device : ``str``
        Device to use for training. Default: ``cpu``.
    verbose : ``int``
        Whether to print all (``2``), some (``1``), or no messages (``0``).

    Returns
    -------
    Training history dictionary.
    """

    optimizer = torch.optim.Adam(model.parameters(), lr, weight_decay=weight_decay)

    data = torch.as_tensor(data)
    validation_data = torch.as_tensor(validation_data) if validation_data is not None else None

    if context is not None:
        use_context = True
        context = torch.as_tensor(context)
    else:
        use_context = False

    validation_context = torch.as_tensor(validation_context) if validation_context is not None else None

    if validation_data is not None:
        if use_context:
            train_dl = DataLoader(TensorDataset(data, context), batch_size, shuffle)
            val_dl = DataLoader(TensorDataset(validation_data, validation_context), batch_size, shuffle)
        else:
            train_dl = DataLoader(TensorDataset(data), batch_size, shuffle)
            val_dl = DataLoader(TensorDataset(validation_data), batch_size, shuffle)
        validation = True
    else:
        if 0.0 < validation_split < 1.0:
            validation = True
            split = int(data.size()[0] * (1. - validation_split))
            if use_context:
                data, validation_data = data[:split], data[split:]
                context, validation_context = context[:split], context[split:]
                train_dl = DataLoader(TensorDataset(data, context), batch_size, shuffle)
                val_dl = DataLoader(TensorDataset(validation_data, validation_context), batch_size, shuffle)
            else:
                data, validation_data = data[:split], data[split:]
                train_dl = DataLoader(TensorDataset(data), batch_size, shuffle)
                val_dl = DataLoader(TensorDataset(validation_data), batch_size, shuffle)
        else:
            validation = False
            if use_context:
                train_dl = DataLoader(TensorDataset(data, context), batch_size, shuffle)
            else:
                train_dl = DataLoader(TensorDataset(data), batch_size, shuffle)

    history = dict()  # Collects per-epoch loss
    history['loss'] = []
    history['val_loss'] = []

    if not validation:
        monitor = 'loss'
    best_epoch = 0
    best_loss = np.inf
    best_model = copy.deepcopy(model.state_dict())

    start_time_sec = time.time()

    for epoch in range(epochs):

        # --- TRAIN AND EVALUATE ON TRAINING SET -----------------------------
        model.train()
        train_loss = 0.0

        for batch in train_dl:

            optimizer.zero_grad()
            loss = compute_loss(model, batch, device, laplace_prior_scale, gaussian_prior_scale, use_context)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
            optimizer.step()

            # train_loss += loss.data.item() * x.size(0)
            train_loss += loss.data.item()

        train_loss = train_loss / len(train_dl.dataset)

        history['loss'].append(train_loss)

        # --- EVALUATE ON VALIDATION SET -------------------------------------
        model.eval()
        if validation:
            val_loss = 0.0

            for batch in val_dl:
                with torch.no_grad():
                    loss = compute_loss(model, batch, device, laplace_prior_scale, gaussian_prior_scale, use_context)
                # val_loss += loss.data.item() * x.size(0)
                val_loss += loss.data.item()

            val_loss = val_loss / len(val_dl.dataset)

            history['val_loss'].append(val_loss)

        if verbose > 1:
            try:
                print('Epoch %3d/%3d, train loss: %5.2f, val loss: %5.2f' % (epoch + 1, epochs, train_loss, val_loss))
            except:  # TODO specify type of exception
                print('Epoch %3d/%3d, train loss: %5.2f' % (epoch + 1, epochs, train_loss))

        # Monitor loss
        if history[monitor][-1] < best_loss:
            best_loss = history[monitor][-1]
            best_epoch = epoch
            best_model = copy.deepcopy(model.state_dict())

        if epoch - best_epoch >= patience:
            model.load_state_dict(best_model)
            if verbose > 0:
                print('Finished early after %3d epochs' % best_epoch)
                print('Best loss achieved %5.2f' % best_loss)
            break

    # END OF TRAINING LOOP

    if verbose > 0:
        end_time_sec = time.time()
        total_time_sec = end_time_sec - start_time_sec
        time_per_epoch_sec = total_time_sec / epochs
        print()
        print('Time total:     %5.2f sec' % total_time_sec)
        print('Time per epoch: %5.2f sec' % time_per_epoch_sec)

    return history


class Flow:
    r"""
    Normalising Flow class.
    This class implements forward and inverse passes, log density evaluation, sampling and model fitting
    irrespective of the kind of flow used.

    Parameters
    ----------
    ndim : ``int``
        Number of dimensions.
    flow_config : ``dict`` or ``None``
        Configuration of the flow. If ``None`` the default configuration used is ``dict(n_blocks=6,
        hidden_size= 3 * ndim, n_hidden=1, batch_norm=True, activation='relu', input_order='sequential',
        flow_type='maf')``
    train_config : ``dict`` or ``None``
        Training configuration for the flow. If ``None`` the default configuration used is
        ``dict(validation_split=0.2, epochs=1000, batch_size=nparticles, patience=30, monitor='val_loss',
        shuffle=True, lr=[1e-2, 1e-3, 1e-4, 1e-5], weight_decay=1e-8, clip_grad_norm=1.0, laplace_prior_scale=0.2,
        gaussian_prior_scale=None, device='cpu', verbose=0)``
    """
    def __init__(self, ndim: int, flow_config: dict = None, train_config: dict = None):
        
        if ndim == 1:
            raise ValueError(f"1D data is not supported. Please provide data with ndim >= 2.")

        self.ndim = ndim
        self.flow_config = flow_config  # TODO remove this if unused.
        self.train_config = train_config  # TODO remove this if unused.

        self.default_config = dict(
            n_blocks=6,
            hidden_size=3 * self.ndim,
            n_hidden=1,
            batch_norm=True,
            activation='relu',
            input_order='sequential',
            flow_type='maf'
        )
        self.flow = self.create(flow_config)

    def validate_config(self, config: dict):
        """
        Raise a ValueError if the flow config dictionary contains invalid inputs.

        Parameters
        ----------
        config: ``dict``
            dictionary with key-value pairs to be passed to constructors of MAF or RealNVP.
        """
        allowed_keys = list(self.default_config.keys())
        for k in config.keys():
            if k not in allowed_keys:
                raise ValueError(f"Unrecognized config key: {k}. Allowed keys are: {allowed_keys}")

    def create(self, config: dict = None):
        """
        Create a normalizing flow based on the desired flow configuration.

        Parameters
        ----------
        config: ``dict``
            dictionary with key-value pairs to be passed to constructors of MAF or RealNVP.

        Returns
        -------
        NF : ``MAF or RealNVP``
            Normalising flow instance.
        """
        if config is None:
            config = dict()
        self.validate_config(config)
        config = {**self.default_config, **config}  # Overwrite keys in default_config and add new ones

        if config['flow_type'].lower() == 'maf':
            return MAF(input_size=self.ndim, cond_label_size=None, **config)
        elif config['flow_type'].lower() == 'realnvp':
            return RealNVP(input_size=self.ndim, cond_label_size=None, **config)
        else:
            raise ValueError(f"Unsupported flow type: {config['flow_type']}. Please use one of ['maf', 'realnvp'].")

    def fit(self, x: torch.Tensor) -> Dict[str, List]:
        """
        Train the normalising flow on the provided data.

        Parameters
        ----------
        x : ``torch.Tensor``
            Training data with shape (n_samples, n_dimensions).
        
        Returns
        -------
        history: ``dict``
            Training history containing logged scalars from model fitting. Each scalar has its own key and value which
            is a list of per-epoch values.
        """
        x = torch_double_to_float(x)

        default_train_config = dict(
            validation_split=0.2,
            epochs=1000,
            batch_size=x.shape[0],
            patience=30,
            monitor='val_loss',
            shuffle=True,
            lr=[1e-2, 1e-3, 1e-4, 1e-5],
            weight_decay=1e-8,
            clip_grad_norm=1.0,
            laplace_prior_scale=0.2,
            gaussian_prior_scale=None,
            device='cpu',
            verbose=0
        )
        if self.train_config is None:
            self.train_config = dict()
        train_config = {**default_train_config, **self.train_config}

        history = None
        for lr in train_config['lr']:
            train_config_tmp = copy.deepcopy(train_config)
            del train_config_tmp['lr']
            history = fit(self.flow, x, **train_config_tmp, lr=lr)

        return history

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        return self.flow.forward(x)

    def inverse(self, u: torch.Tensor) -> torch.Tensor:
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
        return self.flow.inverse(u)

    def logprob(self, x: torch.Tensor) -> torch.Tensor:  # TODO rename to log_prob
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
        u, log_abs_det_jac = self.flow.forward(x)
        return torch.sum(self.flow.base_dist.log_prob(u) + log_abs_det_jac, dim=1)

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
        u = torch.randn(size, self.ndim)
        x, log_abs_det_jac = self.flow.inverse(u)
        return x, torch.sum(self.flow.base_dist.log_prob(u) + log_abs_det_jac, dim=1)
