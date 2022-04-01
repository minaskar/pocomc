from .maf import MAF
from .train import FlowTrainer
import torch

def FlowGenerator(ndim, flow_config=None):

    default_flow_config = dict(n_blocks=6,
                               hidden_size=3*ndim,
                               n_hidden=1,
                               batch_norm=True,
                               activation='relu',
                               dropout=None,
                               input_order='sequential'
                               )

    if flow_config is None:
        flow_config = default_flow_config

    return MAF(n_blocks=flow_config.get('n_blocks', default_flow_config['n_blocks']),
               input_size=ndim,
               hidden_size=flow_config.get('hidden_size', default_flow_config['hidden_size']),
               n_hidden=flow_config.get('n_hidden', default_flow_config['n_hidden']),
               cond_label_size=None,
               activation=flow_config.get('activation', default_flow_config['activation']),
               dropout=flow_config.get('dropout', default_flow_config['dropout']),
               input_order=flow_config.get('input_order', default_flow_config['input_order']),
               batch_norm=flow_config.get('batch_norm', default_flow_config['batch_norm']))

class Flow:

    def __init__(self, ndim, flow_config=None, train_config=None):
        self.ndim = ndim 
        self.flow_config=flow_config
        self.train_config=train_config

        self.flow = FlowGenerator(ndim, flow_config)

    def fit(self, x):
        return FlowTrainer(self.flow, x, self.train_config)

    def forward(self, x):
        return self.flow.forward(x)
        
    def inverse(self, u):
        return self.flow.inverse(u)

    def logprob(self, x):
        u, logdetJ = self.flow.forward(x)
        return torch.sum(self.flow.base_dist.log_prob(u) + logdetJ, dim=1)