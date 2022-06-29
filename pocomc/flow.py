from .maf import MAF, RealNVP
from .train import FlowTrainer
import torch


class Flow:
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

    def fit(self, x):
        return FlowTrainer(self.flow, x, self.train_config)

    def forward(self, x):
        return self.flow.forward(x)

    def inverse(self, u):
        return self.flow.inverse(u)

    def logprob(self, x):
        u, logdetJ = self.flow.forward(x)
        return torch.sum(self.flow.base_dist.log_prob(u) + logdetJ, dim=1)

    def sample(self, size=1):
        u = torch.randn(size, self.ndim)
        x, logdetJ = self.flow.inverse(u)
        return x, torch.sum(self.flow.base_dist.log_prob(u) + logdetJ, dim=1)
