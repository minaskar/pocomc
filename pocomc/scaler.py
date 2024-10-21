import numpy as np
from scipy.special import erf, erfinv

class MultivariateTransform:
    """
    Class to perform multivariate transformations on data, handling periodic,
    bounded, and unbounded parameters using specified transformations.

    Parameters
    ----------
    bounds : list of tuples
        List of (lower, upper) bounds for each parameter.
    periodic : list of int
        List of indices of periodic parameters.
    transform_type : str, optional
        Type of transformation for bounded parameters ("logit" or "probit").
        Default is "probit".

    Attributes
    ----------
    transformations : list
        List containing the sequence of transformations for each parameter.
    """

    def __init__(self, bounds, periodic=None, transform_type='probit'):
        """
        Initialize the MultivariateTransform.

        Parameters
        ----------
        bounds : list of 2-tuples
            Bounds for each parameter. Each tuple contains (lower, upper).
        periodic : list of int
            Indices of the periodic parameters.
        transform_type : str, optional
            Type of transformation to use for bounded parameters. Options are "logit" and "probit".
            Default is "logit".
        """
        self.bounds = bounds
        if periodic is None:
            periodic = []
        self.periodic = periodic
        self.transform_type = transform_type
        self.transformations = []  # List of lists of transformations for each parameter

        for i, (lower, upper) in enumerate(bounds):
            transformations = []

            # Check if the parameter is periodic
            if i in periodic:
                # Apply PeriodicTranslation
                transformations.append(PeriodicTranslation(lower=lower, upper=upper))

            # Determine if the parameter is bounded
            finite_lower = np.isfinite(lower)
            finite_upper = np.isfinite(upper)

            if finite_lower and finite_upper:
                # Both bounds are finite
                # Apply BoundedToUnboundedTransform1D
                transformations.append(BoundedToUnboundedTransform1D(
                    lower=lower, upper=upper, transform=self.transform_type))
            elif finite_lower and not finite_upper:
                # Lower-bounded parameter
                transformations.append(BoundedToUnboundedTransform1D(
                    lower=lower, upper=np.inf, transform='lower'))
            elif not finite_lower and finite_upper:
                # Upper-bounded parameter
                transformations.append(BoundedToUnboundedTransform1D(
                    lower=-np.inf, upper=upper, transform='upper'))
            # else:
                # Unbounded parameter, no BoundedToUnboundedTransform1D needed

            # Finally, apply AffineTransform1D
            transformations.append(AffineTransform1D())

            # Store the transformations for this parameter
            self.transformations.append(transformations)

    def fit(self, X):
        """
        Fit the transformations to the data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_parameters)
            The data to fit.
        """
        X = np.asarray(X, dtype=np.float64)
        n_samples, n_parameters = X.shape

        if n_parameters != len(self.bounds):
            raise ValueError("Number of parameters in X does not match the number of bounds.")

        # For each parameter
        for i in range(n_parameters):
            xi = X[:, i]
            # For each transformation
            for transformation in self.transformations[i]:
                # Fit the transformation on xi
                transformation.fit(xi)
                # Apply the transformation to xi to get xi for the next transformation
                xi, _ = transformation.forward(xi)

    def forward(self, X):
        """
        Apply the forward transformations to the data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_parameters)
            The data to transform.

        Returns
        -------
        U : ndarray of shape (n_samples, n_parameters)
            The transformed data.
        log_det_J : ndarray of shape (n_samples,)
            The sum of the log determinants of the Jacobians for each sample.
        """
        X = np.asarray(X, dtype=np.float64)
        n_samples, n_parameters = X.shape

        if n_parameters != len(self.bounds):
            raise ValueError("Number of parameters in X does not match the number of bounds.")

        U = np.zeros_like(X)
        log_det_J = np.zeros(n_samples)

        # For each parameter
        for i in range(n_parameters):
            xi = X[:, i]
            total_log_det = np.zeros(n_samples)
            # For each transformation
            for transformation in self.transformations[i]:
                xi, log_det = transformation.forward(xi)
                total_log_det += log_det
            U[:, i] = xi
            log_det_J += total_log_det

        return U, log_det_J

    def inverse(self, U):
        """
        Apply the inverse transformations to the data.

        Parameters
        ----------
        U : array-like of shape (n_samples, n_parameters)
            The transformed data.

        Returns
        -------
        X : ndarray of shape (n_samples, n_parameters)
            The original data.
        log_det_J : ndarray of shape (n_samples,)
            The sum of the log determinants of the Jacobians for each sample.
        """
        U = np.asarray(U, dtype=np.float64)
        n_samples, n_parameters = U.shape

        if n_parameters != len(self.bounds):
            raise ValueError("Number of parameters in U does not match the number of bounds.")

        X = np.zeros_like(U)
        log_det_J = np.zeros(n_samples)

        # For each parameter
        for i in range(n_parameters):
            ui = U[:, i]
            total_log_det = np.zeros(n_samples)
            # For each transformation in reverse order
            for transformation in reversed(self.transformations[i]):
                ui, log_det = transformation.inverse(ui)
                total_log_det += log_det
            X[:, i] = ui
            log_det_J += total_log_det

        return X, log_det_J
    

class PeriodicTranslation:
    def __init__(self, lower, upper):
        """
        Initialize the PeriodicTranslation with lower and upper bounds.

        Parameters:
        - lower (float): The lower bound of the domain.
        - upper (float): The upper bound of the domain.
        """
        if upper <= lower:
            raise ValueError("Upper bound must be greater than lower bound.")
        self.lower = lower
        self.upper = upper
        self.width = upper - lower
        self.shift = None  # To be set in fit

    def fit(self, x):
        """
        Fit the translation parameter to center the distribution of x
        at the midpoint between lower and upper bounds under periodic conditions.

        Parameters:
        - x (array-like): 1D array of data points within [lower, upper].
        """
        x = np.asarray(x)
        if x.ndim != 1:
            raise ValueError("Input x must be a 1D array.")
        if np.any(x < self.lower) or np.any(x >= self.upper):
            raise ValueError("All elements of x must be within [lower, upper).")

        # Map x to angles in [0, 2*pi)
        angles = 2 * np.pi * (x - self.lower) / self.width
        # Compute mean angle using complex representation
        mean_complex = np.mean(np.exp(1j * angles))
        mean_angle = np.angle(mean_complex)
        # Ensure mean_angle is in [0, 2*pi)
        if mean_angle < 0:
            mean_angle += 2 * np.pi

        # Desired mean angle is pi (midpoint)
        desired_mean_angle = np.pi

        # Compute the smallest angle difference
        delta_angle = desired_mean_angle - mean_angle
        # Wrap delta_angle to [-pi, pi)
        delta_angle = (delta_angle + np.pi) % (2 * np.pi) - np.pi

        # Convert angle shift to x shift
        self.shift = (delta_angle * self.width) / (2 * np.pi)

    def forward(self, x):
        """
        Apply the forward transformation by shifting x.

        Parameters:
        - x (array-like): 1D array of data points within [lower, upper).

        Returns:
        - x_transformed (np.ndarray): Shifted data within [lower, upper).
        - log_det (float): Log determinant of the transformation (0.0).
        """
        if self.shift is None:
            raise ValueError("The fit method must be called before forward transformation.")
        
        x = np.asarray(x)
        if x.ndim != 1:
            raise ValueError("Input x must be a 1D array.")
        if np.any(x < self.lower) or np.any(x >= self.upper):
            raise ValueError("All elements of x must be within [lower, upper).")

        # Apply shift and wrap around using modulo
        x_shifted = self.lower + ((x + self.shift - self.lower) % self.width)
        log_det = np.zeros(len(x_shifted))  # Log determinant of translation is zero
        return x_shifted, log_det

    def inverse(self, x):
        """
        Apply the inverse transformation by shifting x back.

        Parameters:
        - x (array-like): 1D array of data points within [lower, upper).

        Returns:
        - x_inverted (np.ndarray): Inversely shifted data within [lower, upper).
        - log_det (float): Log determinant of the inverse transformation (0.0).
        """
        if self.shift is None:
            raise ValueError("The fit method must be called before inverse transformation.")
        
        x = np.asarray(x)
        if x.ndim != 1:
            raise ValueError("Input x must be a 1D array.")
        if np.any(x < self.lower) or np.any(x >= self.upper):
            raise ValueError("All elements of x must be within [lower, upper).")

        # Apply inverse shift and wrap around using modulo
        x_inverted = self.lower + ((x - self.shift - self.lower) % self.width)
        log_det = np.zeros(len(x_inverted))  # Log determinant of translation is zero
        return x_inverted, log_det
    



class BoundedToUnboundedTransform1D:
    """
    Class to transform a bounded scalar parameter to an unbounded space using
    logit, probit, lower, or upper transformations.

    Parameters
    ----------
    lower : float
        Lower bound of the parameter.
    upper : float
        Upper bound of the parameter.
    transform : str, optional
        Type of transformation to apply. Options are "logit", "probit", "lower", and "upper".
        Default is "logit".

    Attributes
    ----------
    lower : float
        Lower bound of the parameter.
    upper : float
        Upper bound of the parameter.
    transform : str
        Chosen transformation type.
    """

    def __init__(self, lower: float, upper: float, transform: str = "logit"):
        """
        Initialize the BoundedToUnboundedTransform1D.

        Parameters
        ----------
        lower : float
            Lower bound of the parameter.
        upper : float
            Upper bound of the parameter.
        transform : str, optional
            Type of transformation to apply. Options are "logit", "probit", "lower", and "upper".
            Default is "logit".
        """
        if not isinstance(lower, (int, float)):
            raise TypeError("Lower bound must be a numeric type.")
        if not isinstance(upper, (int, float)):
            raise TypeError("Upper bound must be a numeric type.")
        if transform not in ["logit", "probit", "lower", "upper"]:
            raise ValueError('Invalid transform type. Choose "logit", "probit", "lower", or "upper".')

        # For 'logit' and 'probit', both bounds must be finite
        if transform in ["logit", "probit"]:
            if not np.isfinite(lower) or not np.isfinite(upper):
                raise ValueError(f'Both lower and upper bounds must be finite for "{transform}" transformation.')
            if upper <= lower:
                raise ValueError("Upper bound must be greater than lower bound for 'logit' and 'probit' transformations.")
        elif transform == "lower":
            if not np.isfinite(lower):
                raise ValueError('Lower bound must be finite for "lower" transformation.')
            if upper <= lower:
                raise ValueError("Upper bound must be greater than lower bound for 'lower' transformation.")
        elif transform == "upper":
            if not np.isfinite(upper):
                raise ValueError('Upper bound must be finite for "upper" transformation.')
            if upper <= lower:
                raise ValueError("Upper bound must be greater than lower bound for 'upper' transformation.")

        self.lower = lower
        self.upper = upper
        self.transform = transform

    def fit(self, x: np.ndarray):
        """
        Fit the transformation by validating input data.

        Parameters
        ----------
        x : np.ndarray
            1D array of data points within [lower, upper] for "logit" and "probit",
            [lower, ∞) for "lower", and (-∞, upper] for "upper".

        Raises
        ------
        ValueError
            If any element in x is outside the required bounds.
        """
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 1:
            raise ValueError("Input x must be a 1D array.")

        if self.transform in ["logit", "probit"]:
            if np.any(x < self.lower) or np.any(x > self.upper):
                raise ValueError("All elements of x must be within [lower, upper].")
        elif self.transform == "lower":
            if np.any(x < self.lower):
                raise ValueError("All elements of x must be within [lower, ∞).")
        elif self.transform == "upper":
            if np.any(x > self.upper):
                raise ValueError("All elements of x must be within (-∞, upper].")

    def forward(self, x: np.ndarray):
        """
        Apply the forward transformation (bounded to unbounded).

        Parameters
        ----------
        x : np.ndarray
            1D array of data points within the required bounds.

        Returns
        -------
        u : np.ndarray
            Transformed data in unbounded space.
        log_det_J : np.ndarray
            Log determinant of the Jacobian matrix for each data point.
        """
        # Fit is required to validate inputs
        self.fit(x)

        x = np.asarray(x, dtype=np.float64)

        if self.transform == "logit":
            p = (x - self.lower) / (self.upper - self.lower)
            # Clip p to avoid numerical issues
            eps = 1e-12
            p = np.clip(p, eps, 1 - eps)

            u = np.log(p / (1 - p))
            # Log determinant: d(u)/d(x) = 1 / (p * (1 - p)) * (1 / (upper - lower))
            # Thus, log|du/dx| = -log(p * (1 - p)) - log(upper - lower)
            log_det_J = -np.log(p * (1 - p)) - np.log(self.upper - self.lower)

        elif self.transform == "probit":
            p = (x - self.lower) / (self.upper - self.lower)
            # Clip p to avoid numerical issues
            eps = 1e-12
            p = np.clip(p, eps, 1 - eps)

            u = erfinv(2 * p - 1) * np.sqrt(2)
            # Log determinant: du/dx = sqrt(2pi) * exp(u^2 / 2) / (upper - lower)
            # Thus, log|du/dx| = 0.5 * log(2pi) + 0.5 * u^2 - log(upper - lower)
            log_det_J = 0.5 * np.log(2 * np.pi) + 0.5 * u**2 - np.log(self.upper - self.lower)

        elif self.transform == "lower":
            # Ensure x > lower to avoid log(0)
            eps = 1e-12
            x_shifted = x - self.lower
            x_shifted = np.maximum(x_shifted, eps)

            u = np.log(x_shifted)
            # Log determinant: du/dx = 1 / (x - lower)
            # Thus, log|du/dx| = -log(x - lower)
            log_det_J = -np.log(x_shifted)

        elif self.transform == "upper":
            # Ensure upper - x > 0 to avoid log(0)
            eps = 1e-12
            x_shifted = self.upper - x
            x_shifted = np.maximum(x_shifted, eps)

            u = np.log(x_shifted)
            # Log determinant: du/dx = -1 / (upper - x)
            # Thus, log|du/dx| = -log(upper - x)
            log_det_J = -np.log(x_shifted)

        else:
            raise ValueError('Invalid transform type. Choose "logit", "probit", "lower", or "upper".')

        return u, log_det_J

    def inverse(self, u: np.ndarray):
        """
        Apply the inverse transformation (unbounded to bounded).

        Parameters
        ----------
        u : np.ndarray
            1D array of data points in the unbounded space.

        Returns
        -------
        x : np.ndarray
            Transformed data within the required bounds.
        log_det_J : np.ndarray
            Log determinant of the Jacobian matrix for each data point.
        """
        u = np.asarray(u, dtype=np.float64)

        if self.transform == "logit":
            p = 1 / (1 + np.exp(-u))
            x = self.lower + p * (self.upper - self.lower)
            # Log determinant: d(x)/du = p * (1 - p) * (upper - lower)
            # Thus, log|dx/du| = log(p * (1 - p)) + log(upper - lower)
            log_det_J = np.log(p * (1 - p)) + np.log(self.upper - self.lower)

        elif self.transform == "probit":
            p = (erf(u / np.sqrt(2)) + 1) / 2
            x = self.lower + p * (self.upper - self.lower)
            # Log determinant: dx/du = (upper - lower) * (exp(-u^2 / 2) / sqrt(2*pi))
            # Thus, log|dx/du| = -0.5 * u^2 - 0.5 * log(2*pi) + log(upper - lower)
            log_det_J = -0.5 * u**2 - 0.5 * np.log(2 * np.pi) + np.log(self.upper - self.lower)

        elif self.transform == "lower":
            x = self.lower + np.exp(u)
            # Log determinant: dx/du = exp(u)
            # Thus, log|dx/du| = u
            log_det_J = u

        elif self.transform == "upper":
            x = self.upper - np.exp(u)
            # Log determinant: dx/du = -exp(u)
            # Thus, log|dx/du| = u
            log_det_J = u

        else:
            raise ValueError('Invalid transform type. Choose "logit", "probit", "lower", or "upper".')

        return x, log_det_J


class AffineTransform1D:
    """
    Class to perform 1D affine transformations on data.

    The affine transformation is defined as:
        u = (x - mean) / std

    The inverse transformation is:
        x = u * std + mean

    Parameters
    ----------
    None

    Attributes
    ----------
    mean_ : float
        Mean of the data. Set after fitting.
    std_ : float
        Standard deviation of the data. Set after fitting.
    """

    def __init__(self):
        """
        Initialize the AffineTransform1D.

        No arguments are required during initialization.
        """
        self.mean_ = None  # Mean of the data
        self.std_ = None   # Standard deviation of the data

    def fit(self, x: np.ndarray):
        """
        Fit the affine transformation parameters based on the input data.

        Parameters
        ----------
        x : np.ndarray
            1D array of data points.

        Raises
        ------
        ValueError
            If input x is not a 1D array.
            If scaling is enabled but the standard deviation is zero.
        """
        x = np.asarray(x, dtype=np.float64)

        if x.ndim != 1:
            raise ValueError("Input x must be a 1D array.")

        self.mean_ = np.mean(x)
        self.std_ = np.std(x)

        if self.std_ == 0:
            raise ValueError("Standard deviation of input x is zero. Cannot scale.")

    def forward(self, x: np.ndarray):
        """
        Apply the forward affine transformation to the data.

        Parameters
        ----------
        x : np.ndarray
            1D array of data points to transform.

        Returns
        -------
        u : np.ndarray
            Transformed data.
        log_det_J : float
            Log determinant of the Jacobian matrix.

        Raises
        ------
        ValueError
            If the transformation parameters have not been fitted.
            If input x is not a 1D array.
        """
        if self.mean_ is None or self.std_ is None:
            raise ValueError("The fit method must be called before applying the transformation.")

        x = np.asarray(x, dtype=np.float64)

        if x.ndim != 1:
            raise ValueError("Input x must be a 1D array.")

        u = (x - self.mean_) / self.std_
        log_det_J = -np.log(self.std_) * np.ones(len(u))

        return u, log_det_J

    def inverse(self, u: np.ndarray):
        """
        Apply the inverse affine transformation to the data.

        Parameters
        ----------
        u : np.ndarray
            1D array of transformed data points to invert.

        Returns
        -------
        x : np.ndarray
            Recovered original data.
        log_det_J : float
            Log determinant of the Jacobian matrix of the inverse transformation.

        Raises
        ------
        ValueError
            If the transformation parameters have not been fitted.
            If input u is not a 1D array.
        """
        if self.mean_ is None or self.std_ is None:
            raise ValueError("The fit method must be called before applying the transformation.")

        u = np.asarray(u, dtype=np.float64)

        if u.ndim != 1:
            raise ValueError("Input u must be a 1D array.")

        x = u * self.std_ + self.mean_
        log_det_J = np.log(self.std_) * np.ones(len(u))

        return x, log_det_J