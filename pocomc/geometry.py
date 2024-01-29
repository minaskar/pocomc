import numpy as np

from .student import fit_mvstud
from .tools import systematic_resample

class Geometry:
    """
    Geometry class for the POCOMC algorithm.

    Attributes
    ----------
    normal_mean : array_like
        Mean of the normal distribution.
    normal_cov : array_like
        Covariance matrix of the normal distribution.
    t_mean : array_like
        Mean of the t distribution.
    t_cov : array_like
        Covariance matrix of the t distribution.
    t_nu : float
        Degrees of freedom of the t distribution.
    """

    def __init__(self):
        self.normal_mean = None
        self.normal_cov = None
        self.t_mean = None
        self.t_cov = None
        self.t_nu = None

    def fit(self, theta, weights=None):
        """

        Parameters
        ----------
        theta : array_like
            Array of samples.
        weights : array_like, optional
            Array of weights. The default is None.
        """
        
        # Learn normal distribution
        if weights is None:
            self.normal_mean = np.mean(theta, axis=0)
            self.normal_cov = np.cov(theta.T)
        else:
            self.normal_mean = np.average(theta, axis=0, weights=weights)
            self.normal_cov = np.cov(theta.T, aweights=weights)

        # Learn t distribution
        if weights is not None:
            idx_resampled = systematic_resample(len(theta), weights=weights)
            theta_resampled = theta[idx_resampled]
            self.t_mean, self.t_cov, self.t_nu = fit_mvstud(theta_resampled)
        else:
            self.t_mean, self.t_cov, self.t_nu = fit_mvstud(theta)

        if ~np.isfinite(self.t_nu):
            self.t_nu = 1e6