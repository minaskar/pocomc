import numpy as np
from torch import inverse

class StandardScaler:

    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, x):
        self.mean = np.mean(x, axis=0)
        self.std = np.std(x, axis=0) 

    def transform(self, x):
        return (x - self.mean) / self.std

    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)

    def inverse_transform(self, u):
        return self.mean + u * self.std

    def refit(self, u):
        x = self.inverse_transform(u)
        self.fit(x)

    def refit_transform(self, u):
        x = self.inverse_transform(u)
        self.fit(x)
        return self.transform(x)


class IdentityScaler:

    def __init__(self):
        pass

    def fit(self, x):
        pass 

    def transform(self, x):
        return x

    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)

    def inverse_transform(self, u):
        return u

    def refit(self, u):
        x = self.inverse_transform(u)
        self.fit(x)

    def refit_transform(self, u):
        x = self.inverse_transform(u)
        self.fit(x)
        return self.transform(x)