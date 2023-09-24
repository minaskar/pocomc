import numpy as np

class Prior:

    def __init__(self, dists=None):
        self.dists = dists

    def logpdf(self, x):
        logp = np.zeros(len(x))
        for i, dist in enumerate(self.dists): 
            logp += dist.logpdf(x[:,i])
        return logp
    
    def rvs(self, size=1):
        samples = []
        for dist in self.dists:
            samples.append(dist.rvs(size=size))
        return np.transpose(samples)
    
    @property
    def bounds(self):
        bounds = []
        for dist in self.dists:
            bounds.append(dist.support())
        return np.array(bounds)
    
    @property
    def dim(self):
        return len(self.dists)