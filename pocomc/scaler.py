from typing import Union, List

import numpy as np
from scipy.special import erf, erfinv

from .input_validation import assert_array_float, assert_array_within_interval

class Reparameterize:
    """
    Class that reparameterises the model using change-of-variables parameter transformations.

    Parameters
    ----------
    n_dim : ``int``
        Dimensionality of sampling problem
    bounds : ``np.ndarray`` or ``list`` or ``None``
        Parameter bounds
    periodic : ``list``
        List of indices corresponding to parameters with periodic boundary conditions
    reflective : ``list``
        List of indices corresponding to parameters with reflective boundary conditions
    transform : ``str``
        Type of transform to use for bounded parameters. Options are ``"probit"``
        (default) and ``"logit"``.
    scale : ``bool``
        Rescale parameters to zero mean and unit variance (default is true)
    diagonal : ``bool``
        Use diagonal transformation (i.e. ignore covariance) (default is true)

    Examples
    --------
    >>> import numpy as np
    >>> from pocomc.reparameterize import Reparameterize
    >>> bounds = np.array([[0, 1], [0, 1]])
    >>> reparam = Reparameterize(2, bounds)
    >>> x = np.array([[0.5, 0.5], [0.5, 0.5]])
    >>> reparam.forward(x)
    array([[0., 0.],
           [0., 0.]])
    >>> u = np.array([[0, 0], [0, 0]])
    >>> reparam.inverse(u)
    (array([[0.5, 0.5],
           [0.5, 0.5]]), array([0., 0.]))
    """
    def __init__(self,
                 n_dim: int,
                 bounds: Union[np.ndarray, list] = None,
                 periodic: List[int] = None,
                 reflective: List[int] = None,
                 transform: str = "probit",
                 scale: bool = True,
                 diagonal: bool = True):

        self.ndim = n_dim

        if bounds is None:
            bounds = np.full((self.ndim, 2), np.inf)
        elif len(bounds) == 2 and not np.shape(bounds) == (2, 2):
            bounds = np.tile(np.array(bounds, dtype=np.float32).reshape(2, 1), self.ndim).T
        assert_array_float(bounds)

        self.low = bounds.T[0]
        self.high = bounds.T[1]

        self.periodic = periodic
        self.reflective = reflective

        if transform not in ["logit", "probit"]:
            raise ValueError("Please provide a valid transformation function (e.g. logit or probit)")
        else:
            self.transform = transform

        self.mu = None
        self.sigma = None
        self.cov = None
        self.L = None
        self.L_inv = None
        self.log_det_L = None
        self.scale = scale
        self.diagonal = diagonal

        self._create_masks()

    def apply_boundary_conditions_x(self, x: np.ndarray):
        """
        Apply boundary conditions (i.e. periodic or reflective) to input ``x``.
        The first kind include phase parameters that might be periodic
        e.g. on a range ``[0,2*np.pi]``. The latter can arise in cases
        where parameters are ratios where ``a/b`` and  ``b/a`` are equivalent.

        Parameters
        ----------
        x : np.ndarray
            Input array
        
        Returns
        -------
        Transformed input
        """
        if (self.periodic is None) and (self.reflective is None):
            return x
        elif self.periodic is None:
            return self._apply_reflective_boundary_conditions_x(x)
        elif self.reflective is None:
            return self._apply_periodic_boundary_conditions_x(x)
        else:
            return self._apply_reflective_boundary_conditions_x(self._apply_periodic_boundary_conditions_x(x))

    def _apply_periodic_boundary_conditions_x(self, x: np.ndarray):
        """
        Apply periodic boundary conditions to input ``x``.
        This can be useful for phase parameters that might be periodic
        e.g. on a range ``[0,2*np.pi]``
        
        Parameters
        ----------
        x : np.ndarray
            Input array
        
        Returns
        -------
        Transformed input.
        """
        if self.periodic is not None:
            x = x.copy()
            for i in self.periodic:
                for j in range(len(x)):
                    while x[j, i] > self.high[i]:
                        x[j, i] = self.low[i] + x[j, i] - self.high[i]
                    while x[j, i] < self.low[i]:
                        x[j, i] = self.high[i] + x[j, i] - self.low[i]
        return x

    def _apply_reflective_boundary_conditions_x(self, x: np.ndarray):
        """
        Apply reflective boundary conditions to input ``x``. This can arise in cases
        where parameters are ratios where ``a/b`` and  ``b/a`` are equivalent.
        
        Parameters
        ----------
        x : np.ndarray
            Input array
        
        Returns
        -------
        Transformed input.
        """
        if self.reflective is not None:
            x = x.copy()
            for i in self.reflective:
                for j in range(len(x)):
                    while x[j, i] > self.high[i]:
                        x[j, i] = self.high[i] - x[j, i] + self.high[i]
                    while x[j, i] < self.low[i]:
                        x[j, i] = self.low[i] + self.low[i] - x[j, i]

        return x

    def fit(self, x: np.ndarray):
        """
        Learn mean and standard deviation using for rescaling.
        
        Parameters
        ----------
        x : np.ndarray
            Input data used for training.
        """
        assert_array_within_interval(x, self.low, self.high)

        u = self._forward(x)
        self.mu = np.mean(u, axis=0)
        if self.diagonal:
            self.sigma = np.std(u, axis=0)
        else:
            self.cov = np.cov(u.T)
            self.L = np.linalg.cholesky(self.cov)
            self.L_inv = np.linalg.inv(self.L)
            self.log_det_L = np.linalg.slogdet(self.L)[1]

    def forward(self, x: np.ndarray, check_input=True):
        """
        Forward transformation (both logit/probit for bounds and affine for all parameters).

        Parameters
        ----------
        x : np.ndarray
            Input data
        check_input : bool
            Check if input is within bounds (default: True)
        Returns
        -------
        u : np.ndarray
            Transformed input data
        """
        if check_input:
            assert_array_within_interval(x, self.low, self.high)

        u = self._forward(x)
        if self.scale:
            u = self._forward_affine(u)

        return u

    def inverse(self, u: np.ndarray):
        """
        Inverse transformation (both logit^-1/probit^-1 for bounds and affine for all parameters).

        Parameters
        ----------
        u : np.ndarray
            Input data
        Returns
        -------
        x : np.ndarray
            Transformed input data
        log_det_J : np.array
            Logarithm of determinant of Jacobian matrix transformation.
        """
        if self.scale:
            x, log_det_J = self._inverse_affine(u)
            x, log_det_J_prime = self._inverse(x)
            log_det_J += log_det_J_prime
        else:
            x, log_det_J = self._inverse(u)

        return x, log_det_J

    def _forward(self, x: np.ndarray):
        """
        Forward transformation (only logit/probit for bounds).

        Parameters
        ----------
        x : np.ndarray
            Input data
        Returns
        -------
        u : np.ndarray
            Transformed input data
        """
        u = np.empty(x.shape)
        u[:, self.mask_none] = self._forward_none(x)
        u[:, self.mask_left] = self._forward_left(x)
        u[:, self.mask_right] = self._forward_right(x)
        u[:, self.mask_both] = self._forward_both(x)

        return u

    def _inverse(self, u: np.ndarray):
        """
        Inverse transformation (only logit^-1/probit^-1 for bounds).

        Parameters
        ----------
        u : np.ndarray
            Input data
        Returns
        -------
        x : np.ndarray
            Transformed input data
        log_det_J : np.array
            Logarithm of determinant of Jacobian matrix transformation.
        """
        x = np.empty(u.shape)
        J = np.empty(u.shape)

        x[:, self.mask_none], J[:, self.mask_none] = self._inverse_none(u)
        x[:, self.mask_left], J[:, self.mask_left] = self._inverse_left(u)
        x[:, self.mask_right], J[:, self.mask_right] = self._inverse_right(u)
        x[:, self.mask_both], J[:, self.mask_both] = self._inverse_both(u)

        log_det_J = np.sum(J, axis=1)

        return x, log_det_J

    def _forward_affine(self, x: np.ndarray):
        """
        Forward affine transformation.

        Parameters
        ----------
        x : np.ndarray
            Input data
        Returns
        -------
        Transformed input data
        """
        if self.diagonal:
            return (x - self.mu) / self.sigma
        else:
            return np.array([np.dot(self.L_inv, xi - self.mu) for xi in x])

    def _inverse_affine(self, u: np.ndarray):
        """
        Inverse affine transformation.

        Parameters
        ----------
        u : np.ndarray
            Input data
        Returns
        -------
        x : np.ndarray
            Transformed input data
        J : np.ndarray
            Diagonal of Jacobian matrix.
        """
        if self.diagonal:
            log_det_J = np.sum(np.log(self.sigma))
            return self.mu + self.sigma * u, log_det_J * np.ones(len(u))
        else:
            x = self.mu + np.array([np.dot(self.L, ui) for ui in u])
            return x, self.log_det_L * np.ones(len(u))

    def _forward_left(self, x: np.ndarray):
        """
        Forward transformation for bounded parameters (only low).

        Parameters
        ----------
        x : np.ndarray
            Input data
        Returns
        -------
        Transformed input data
        """
        return np.log(x[:, self.mask_left] - self.low[self.mask_left])

    def _inverse_left(self, u: np.ndarray):
        """
        Inverse transformation for bounded parameters (only low).

        Parameters
        ----------
        u : np.ndarray
            Input data
        Returns
        -------
        x : np.ndarray
            Transformed input data
        J : np.array
            Diagonal of Jacobian matrix.
        """
        p = np.exp(u[:, self.mask_left])

        return np.exp(u[:, self.mask_left]) + self.low[self.mask_left], u[:, self.mask_left]

    def _forward_right(self, x: np.ndarray):
        """
        Forward transformation for bounded parameters (only high).

        Parameters
        ----------
        x : np.ndarray
            Input data
        Returns
        -------
        Transformed input data
        """
        return np.log(self.high[self.mask_right] - x[:, self.mask_right])

    def _inverse_right(self, u: np.ndarray):
        """
        Inverse transformation for bounded parameters (only high).

        Parameters
        ----------
        u : np.ndarray
            Input data
        Returns
        -------
        x : np.ndarray
            Transformed input data
        J : np.array
            Diagonal of Jacobian matrix.
        """

        return self.high[self.mask_right] - np.exp(u[:, self.mask_right]), u[:, self.mask_right]

    def _forward_both(self, x: np.ndarray):
        """
        Forward transformation for bounded parameters (both low and high).

        Parameters
        ----------
        x : np.ndarray
            Input data
        Returns
        -------
        Transformed input data
        """
        p = (x[:, self.mask_both] - self.low[self.mask_both]) / (self.high[self.mask_both] - self.low[self.mask_both])
        np.clip(p, 1e-13, 1.0 - 1e-13)

        if self.transform == "logit":
            u = np.log(p / (1.0 - p))
        elif self.transform == "probit":
            u = np.sqrt(2.0) * erfinv(2.0 * p - 1.0)

        return u

    def _inverse_both(self, u: np.ndarray):
        """
        Inverse transformation for bounded parameters (both low and high).

        Parameters
        ----------
        u : np.ndarray
            Input data
        Returns
        -------
        x : np.ndarray
            Transformed input data
        J : np.array
            Diagonal of Jacobian matrix.
        """
        if self.transform == "logit":
            p = np.exp(-np.logaddexp(0, -u[:, self.mask_both]))
            x = p * (self.high[self.mask_both] - self.low[self.mask_both]) + self.low[self.mask_both]
            J = np.log(self.high[self.mask_both] - self.low[self.mask_both]) + np.log(p) + np.log(1.0 - p)
        elif self.transform == "probit":
            p = ( erf(u[:, self.mask_both] / np.sqrt(2.0)) + 1.0 ) / 2.0
            x = p * (self.high[self.mask_both] - self.low[self.mask_both]) + self.low[self.mask_both]
            J = np.log(self.high[self.mask_both] - self.low[self.mask_both]) + (-u[:, self.mask_both]**2.0 / 2.0) - np.log(np.sqrt(2.0 * np.pi))
        return x, J

    def _forward_none(self, x:np.ndarray):
        """
        Forward transformation for unbounded parameters (this does nothing).

        Parameters
        ----------
        x : np.ndarray
            Input data
        Returns
        -------
        u : np.ndarray
            Transformed input data
        """
        return x[:, self.mask_none]

    def _inverse_none(self, u:np.ndarray):
        """
        Inverse transformation for unbounded parameters (this does nothing).

        Parameters
        ----------
        u : np.ndarray
            Input data
        Returns
        -------
        x : np.ndarray
            Transformed input data
        log_det_J : np.array
            Logarithm of determinant of Jacobian matrix transformation.
        """
        return u[:, self.mask_none], np.zeros(u.shape)[:, self.mask_none]

    def _create_masks(self):
        """
        Create parameter masks for bounded parameters
        """

        self.mask_left = np.zeros(self.ndim, dtype=bool)
        self.mask_right = np.zeros(self.ndim, dtype=bool)
        self.mask_both = np.zeros(self.ndim, dtype=bool)
        self.mask_none = np.zeros(self.ndim, dtype=bool)

        # TODO: Do this more elegantly, it's a shame
        for i in range(self.ndim):
            if not np.isfinite(self.low[i]) and not np.isfinite(self.high[i]):
                self.mask_none[i] = True
                self.mask_left[i] = False
                self.mask_right[i] = False
                self.mask_both[i] = False
            elif not np.isfinite(self.low[i]) and np.isfinite(self.high[i]):
                self.mask_none[i] = False
                self.mask_left[i] = False
                self.mask_right[i] = True
                self.mask_both[i] = False
            elif np.isfinite(self.low[i]) and not np.isfinite(self.high[i]):
                self.mask_none[i] = False
                self.mask_left[i] = True
                self.mask_right[i] = False
                self.mask_both[i] = False
            else:
                self.mask_none[i] = False
                self.mask_left[i] = False
                self.mask_right[i] = False
                self.mask_both[i] = True