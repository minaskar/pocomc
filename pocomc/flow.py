from turtle import forward
from .maf import MAF
from .train import FlowTrainer

def FlowGenerator(ndim, flow_config=None):

    if flow_config is None:
        flow_config = dict(n_blocks=5,
                           hidden_size=100,
                           n_hidden=1,
                           batch_norm=True)

    return MAF(n_blocks=flow_config['n_blocks'],
               input_size=ndim,
               hidden_size=flow_config['hidden_size'],
               n_hidden=flow_config['n_hidden'],
               cond_label_size=None,
               activation='relu',
               input_order='sequential',
               batch_norm=flow_config['batch_norm'])

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