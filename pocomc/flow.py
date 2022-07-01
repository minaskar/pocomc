from .maf import MAF, RealNVP
import torch
from .tools import torch_double_to_float
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import copy
import time


def fit(model,
        data,
        context=None,
        validation_data=None,
        validation_context=None,
        validation_split=0.0,
        epochs=20,
        batch_size=100,
        patience=np.inf,
        monitor='val_loss',
        shuffle=True,
        lr=1e-3,
        weight_decay=1e-8,
        clip_grad_norm=1.0,
        l1=None,
        l2=None,
        device='cpu',
        verbose=2):
    r"""
        Method to fit the normalising flow.

    Parameters
    ----------
    data : ``np.ndarray``
        Samples used for training the flow.
    context : ``np.ndarray`` or None
        Additional samples corresponding to conditional  arguments.
    validation_data : ``np.ndarray`` or None
        Samples used for validating the flow.
    validation_context : ``np.ndarray`` or None
        Addiitional samples correspondingg to conditioonal arguments,
        used for validation.
    validation_split : float
        Percentage of ``data`` to be used for validation.
    epochs : int
        Number of training epochs.
    batch_size : int
        Batch size used for training.
    patience : int
        Number of epochs to wait with no improvement until termination.
    monitor : str
        Which loss to monitor for early stopping (e.g. ``val_loss``, ``loss``),
        default is ``val_loss``.
    shuffle : bool
        Shuffle data before training.
    lr : float
        Learning rate.
    weight_decay : float
        Weight decay parameter.
    clip_grad_norm : float
        Clip large gradients (default is ``0``).
    l1 : float or None
        Laplace prior scale for regularisation.
    l2 : float or None
        Gaussian prior scale for regularisation.
    device : str
        Device to use for training, default is ``cpu``.
    verbose : int
        Whether to print all (``2``), some (``1``), or no messages (``0``).

    Returns
    -------
    Training history dictionary.
    """

    optimizer = torch.optim.Adam(model.parameters(), lr, weight_decay=weight_decay)

    if not isinstance(data, torch.Tensor):
        data = torch.tensor(data, dtype=torch.float32)

    if (validation_data is not None) and (not isinstance(validation_data, torch.Tensor)):
        validation_data = torch.tensor(validation_data, dtype=torch.float32)

    if context is not None:
        use_context = True
        if not isinstance(context, torch.Tensor):
            context = torch.tensor(context, dtype=torch.float32)
    else:
        use_context = False

    if (validation_context is not None) and (not isinstance(validation_context, torch.Tensor)):
        validation_context = torch.tensor(validation_context, dtype=torch.float32)

    if validation_data is not None:

        if use_context:
            train_dl = DataLoader(TensorDataset(data, context), batch_size, shuffle)
            val_dl = DataLoader(TensorDataset(validation_data, validation_context), batch_size, shuffle)
        else:
            train_dl = DataLoader(TensorDataset(data), batch_size, shuffle)
            val_dl = DataLoader(TensorDataset(validation_data), batch_size, shuffle)

        validation = True
    else:
        if validation_split > 0.0 and validation_split < 1.0:
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

    history = {}  # Collects per-epoch loss
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

            if use_context:
                x = batch[0].to(device)
                y = batch[1].to(device)
                loss = -model.log_prob(x, y).sum()
            else:
                x = batch[0].to(device)
                loss = -model.log_prob(x).sum()

            if l1 is not None:
                loss -= model.log_prior(scale=l1, type='Laplace')

            if l2 is not None:
                loss -= model.log_prior(scale=l2, type='Gaussian')

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

                if use_context:
                    x = batch[0].to(device)
                    y = batch[1].to(device)
                    loss = -model.log_prob(x, y).sum()
                else:
                    x = batch[0].to(device)
                    loss = -model.log_prob(x).sum()

                if l1 is not None:
                    loss -= model.log_prior(scale=l1, type='Laplace')

                if l2 is not None:
                    loss -= model.log_prior(scale=l2, type='Gaussian')

                # val_loss += loss.data.item() * x.size(0)
                val_loss += loss.data.item()

            val_loss = val_loss / len(val_dl.dataset)

            history['val_loss'].append(val_loss)

        if verbose > 1:
            try:
                print('Epoch %3d/%3d, train loss: %5.2f, val loss: %5.2f' % \
                      (epoch + 1, epochs, train_loss, val_loss))
            except:
                print('Epoch %3d/%3d, train loss: %5.2f' % \
                      (epoch + 1, epochs, train_loss))

        # Monitor loss
        if history[monitor][-1] < best_loss:
            best_loss = history[monitor][-1]
            best_epoch = epoch
            best_model = copy.deepcopy(model.state_dict())

        if epoch - best_epoch >= patience:
            model.load_state_dict(best_model)
            if verbose > 0:
                print('Finished early after %3d epochs' % (best_epoch))
                print('Best loss achieved %5.2f' % (best_loss))
            break

    # END OF TRAINING LOOP

    if verbose > 0:
        end_time_sec = time.time()
        total_time_sec = end_time_sec - start_time_sec
        time_per_epoch_sec = total_time_sec / epochs
        print()
        print('Time total:     %5.2f sec' % (total_time_sec))
        print('Time per epoch: %5.2f sec' % (time_per_epoch_sec))

    return history


class Flow:
    r"""
    
        Normalising Flow object.

    Parameters
    ----------
    ndim : int
        Number of parameters or dimensions
    flow_config : dict or None
        Configuration of the flow. If ``None`` the default configuration is used ``dict(n_blocks=6,
        hidden_size= 3 * ndim, n_hidden=1, batch_norm=True, activation='relu', input_order='sequential',
        flow_type='maf')``
    train_config : dict or None
        Training confiiguration for the flow. If ``None`` the default configuration is used ``validation_split=0.2,
        epochs=1000, batch_size=nparticles, patience=30, monitor='val_loss', shuffle=True, lr=[1e-2, 1e-3, 1e-4, 1e-5],
        weight_decay=1e-8, clip_grad_norm=1.0, l1=0.2, l2=None, device='cpu', verbose=0``
    """
    def __init__(self, ndim: int, flow_config: dict = None, train_config: dict = None):
        if ndim == 1:
            raise ValueError(f"1D data is not supported. Please provide data with ndim >= 2.")

        self.ndim = ndim
        self.flow_config = flow_config  # TODO do we have to store this?
        self.train_config = train_config

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
        config: dictionary with key-value pairs to be passed onto constructors of MAF or RealNVP.
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
        config: dictionary with key-value pairs to be passed onto constructors of MAF or RealNVP.

        Returns
        -------
        A MAF or RealNVP object with the desired configuration.
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

    def fit(self, x: torch.Tensor):
        """
        Train the normalising flow on the provided data.

        Parameters
        ----------
        x : torch.Tensor
            Training data.
        
        Returns
        -------
        Training history.
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
            l1=0.2,
            l2=None,
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

    def forward(self, x: torch.Tensor):
        """
        Forward transformation.
        
        Parameters
        ----------
        x : torch.Tensor
            Samples to transform.
        Returns
        -------
        Tranformed samples.
        """
        x = torch_double_to_float(x)
        return self.flow.forward(x)

    def inverse(self, u: torch.Tensor):
        """
        Inverse transformation.
        
        Parameters
        ----------
        u : torch.Tensor
            Samples to transform.
        Returns
        -------
        Tranformed samples.
        """
        u = torch_double_to_float(u)
        return self.flow.inverse(u)

    def logprob(self, x: torch.Tensor):
        """
        Log-probability of samples.
        
        Parameters
        ----------
        x : torch.Tensor
            Input samples
        Returns
        -------
        Log-probability of samples.
        """
        x = torch_double_to_float(x)
        u, logdetJ = self.flow.forward(x)
        return torch.sum(self.flow.base_dist.log_prob(u) + logdetJ, dim=1)

    def sample(self, size: int = 1):
        """
        Method that generates random samples.

        Parameters
        ----------
        size : int
            Number of samples to generate.
        Returns
        -------
        samples and respective log-probability values. 
        """
        u = torch.randn(size, self.ndim)
        x, logdetJ = self.flow.inverse(u)
        return x, torch.sum(self.flow.base_dist.log_prob(u) + logdetJ, dim=1)
