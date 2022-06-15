from re import U
import numpy as np

class Reparameterise:

    def __init__(self, bounds, periodic=None, reflective=None, scale=True, diagonal=True):
        self.low = bounds.T[0]
        self.high = bounds.T[1]
        self.ndim = len(self.low)

        self.periodic = periodic
        self.reflective = reflective

        self.mu = None
        self.sigma = None
        self.cov = None
        self.L = None
        self.Linv = None
        self.logdetL = None
        self.scale = scale
        self.diagonal = diagonal

        self._create_masks()

    def apply_boundary_conditions(self, x):
        return self._apply_periodic_boundary_conditions(x)

    def _apply_periodic_boundary_conditions(self, x):

        if self.periodic is not None:
            x = x.copy()
            for i in self.periodic:
                for j in range(len(x)):
                    while x[j,i] > self.high[i]:
                        x[j,i] = self.low[i] + x[j,i] - self.high[i]
                    while x[j,i] < self.low[i]:
                        x[j,i] = self.high[i] + x[j,i] - self.low[i]
        return x

    def _apply_reflective_boundary_conditions(self, x):

        if self.reflective is not None:
            x = x.copy()
            for i in self.reflective:
                for j in range(len(x)):
                    while x[j,i] > self.high[i]:
                        x[j,i] = self.high[i] - x[j,i] + self.high[i]
                    while x[j,i] < self.low[i]:
                        x[j,i] = self.low[i] + self.low[i] - self.x[j,i]

        return x

    def fit(self, x):
        
        u = self._forward(x)
        self.mu = np.mean(u, axis=0)
        if self.diagonal:
            self.sigma = np.std(u, axis=0)
        else:
            self.cov = np.cov(u.T)
            self.L = np.linalg.cholesky(self.cov)
            self.Linv = np.linalg.inv(self.L)
            self.logdetL = np.linalg.slogdet(self.L)[1]

    def forward(self, x):
        
        u = self._forward(x)
        if self.scale:
            u = self._forward_affine(u)

        return u

    def inverse(self, u):
        
        if self.scale:
            x, logdetJ = self._inverse_affine(u)
            x, logdetJ_prime = self._inverse(x)
            logdetJ += logdetJ_prime
        else:
            x, logdetJ = self._inverse(u)

        return x, logdetJ

    def _forward(self, x):

        u = np.empty(x.shape)
        u[:,self.mask_none] = self._forward_none(x)
        u[:,self.mask_left] = self._forward_left(x)
        u[:,self.mask_right] = self._forward_right(x)
        u[:,self.mask_both] = self._forward_both(x)

        return u

    def _inverse(self, u):

        x = np.empty(u.shape)
        J = np.empty(u.shape)

        x[:, self.mask_none], J[:, self.mask_none] = self._inverse_none(u)
        x[:, self.mask_left], J[:, self.mask_left] = self._inverse_left(u)
        x[:, self.mask_right], J[:, self.mask_right] = self._inverse_right(u)
        x[:, self.mask_both], J[:, self.mask_both] = self._inverse_both(u)

        logdetJ = np.array([np.linalg.slogdet(Ji*np.identity(len(Ji)))[1] for Ji in J])

        return x, logdetJ

    def _forward_affine(self, x):
        if self.diagonal:
            return ( x - self.mu ) / self.sigma
        else:
            return np.array([np.dot(self.Linv, xi - self.mu) for xi in x])

    def _inverse_affine(self, u):
        if self.diagonal:
            J = self.sigma
            logdetJ = np.linalg.slogdet(J * np.identity(len(J)))[1]
            return self.mu + self.sigma * u, logdetJ * np.ones(len(u))
        else:
            x = self.mu + np.array([np.dot(self.L,ui) for ui in u])
            return x, self.logdetL * np.ones(len(u))

    def _forward_left(self, x):
        return np.log( x[:,self.mask_left] - self.low[self.mask_left] )

    def _inverse_left(self, u):

        p = np.exp( u[:,self.mask_left] )

        return p + self.low[self.mask_left], p

    def _forward_right(self, x):
        return np.log( self.high[self.mask_right] - x[:,self.mask_right] )

    def _inverse_right(self, u):

        p = np.exp( u[:,self.mask_right] )

        return self.high[self.mask_right] - p, p

    def _forward_both(self, x):

        p = (x[:,self.mask_both] - self.low[self.mask_both]) / (self.high[self.mask_both]-self.low[self.mask_both])

        return np.log( p / ( 1 - p ) )

    def _inverse_both(self, u):

        p = np.exp(-np.logaddexp(0, -u[:,self.mask_both]))

        x = p * (self.high[self.mask_both] - self.low[self.mask_both]) + self.low[self.mask_both]

        J = (self.high[self.mask_both]-self.low[self.mask_both]) * p * (1.0 - p)

        return x, J

    def _forward_none(self, x):
        return x[:,self.mask_none]

    def _inverse_none(self, u):
        return u[:,self.mask_none], np.ones(u.shape)[:,self.mask_none]

    def _create_masks(self):

        self.mask_left = np.zeros(self.ndim, dtype=bool)
        self.mask_right = np.zeros(self.ndim, dtype=bool)
        self.mask_both = np.zeros(self.ndim, dtype=bool)
        self.mask_none = np.zeros(self.ndim, dtype=bool)

        # TODO: Do this more elegantly, it's a shame
        for i in range(self.ndim):
            if np.isnan(self.low[i]) and np.isnan(self.high[i]):
                self.mask_none[i] = True
                self.mask_left[i] = False
                self.mask_right[i] = False
                self.mask_both[i] = False
            elif np.isnan(self.low[i]) and not np.isnan(self.high[i]):
                self.mask_none[i] = False
                self.mask_left[i] = False
                self.mask_right[i] = True
                self.mask_both[i] = False
            elif not np.isnan(self.low[i]) and np.isnan(self.high[i]):
                self.mask_none[i] = False
                self.mask_left[i] = True
                self.mask_right[i] = False
                self.mask_both[i] = False
            else:
                self.mask_none[i] = False
                self.mask_left[i] = False
                self.mask_right[i] = False
                self.mask_both[i] = True