from typing import Any
import numpy as np

class Particles:
    """
    Class to store the particles and their associated weights.

    Parameters
    ----------
    n_particles : int
        Number of particles.
    n_dim : int
        Dimension of the parameter space.
    ess_threshold : float, optional
        Threshold for the effective sample size. If the effective sample size
        is below this threshold, the weights are set to zero. This is useful
        for the case where the effective sample size is very small, but not
        exactly zero, due to numerical errors.
    
    Attributes
    ----------
    n_particles : int
        Number of particles.
    n_dim : int
        Dimension of the parameter space.
    ess_threshold : float, optional
        Threshold for the effective sample size. If the effective sample size
        is below this threshold, the weights are set to zero. This is useful
        for the case where the effective sample size is very small, but not
        exactly zero, due to numerical errors.
    u : numpy.ndarray
        Array of shape (n_particles, n_dim) containing the particles.
    logdetj : numpy.ndarray
        Array of shape (n_particles,) containing the log-determinant of the
        Jacobian of the transformation from the unit hypercube to the
        parameter space.
    logl : numpy.ndarray
        Array of shape (n_particles,) containing the log-likelihoods.
    logp : numpy.ndarray
        Array of shape (n_particles,) containing the log-priors.
    logw : numpy.ndarray
        Array of shape (n_particles,) containing the log-weights.
    iter : numpy.ndarray
        Array of shape (n_particles,) containing the iteration number of each
        particle.
    logz : numpy.ndarray
        Array of shape (n_particles,) containing the log-evidence of each
        particle.
    calls : numpy.ndarray
        Array of shape (n_particles,) containing the number of likelihood
        evaluations of each particle.
    steps : numpy.ndarray
        Array of shape (n_particles,) containing the number of steps of each
        particle.
    efficiency : numpy.ndarray
        Array of shape (n_particles,) containing the efficiency of each
        particle.
    ess : numpy.ndarray
        Array of shape (n_particles,) containing the effective sample size of
        each particle.
    accept : numpy.ndarray
        Array of shape (n_particles,) containing the acceptance rate of each
        particle.
    beta : numpy.ndarray
        Array of shape (n_particles,) containing the inverse temperature of
        each particle.
    """

    def __init__(self, n_particles, n_dim):
        self.n_particles = n_particles
        self.n_dim = n_dim

        self.past = dict(
            u = [],
            x = [],
            logdetj = [],
            logl = [],
            logp = [],
            logw = [],
            blobs = [],
            iter = [],
            logz = [],
            calls = [],
            steps = [],
            efficiency = [],
            ess = [],
            accept = [],
            beta = [],
        )

        self.results_dict = None

    def update(self, data):
        """
        Update the particles with the given data.

        Parameters
        ----------
        data : dict
            Dictionary containing the data to be added to the particles.
        
        Notes
        -----
        The dictionary must contain the following keys:
            u : numpy.ndarray
                Array of shape (n_particles, n_dim) containing the particles.
            logdetj : numpy.ndarray
                Array of shape (n_particles,) containing the log-determinant
                of the Jacobian of the transformation from the unit hypercube
                to the parameter space.
            logl : numpy.ndarray
                Array of shape (n_particles,) containing the log-likelihoods.
            logp : numpy.ndarray
                Array of shape (n_particles,) containing the log-priors.
            logw : numpy.ndarray
                Array of shape (n_particles,) containing the log-weights.
            blobs : numpy.ndarray
                Array of shape (n_particles,) containing the blobs (derived parameters).
            iter : numpy.ndarray
                Array of shape (n_particles,) containing the iteration number
                of each particle.
            logz : numpy.ndarray
                Array of shape (n_particles,) containing the log-evidence of
                each particle.
            calls : numpy.ndarray
                Array of shape (n_particles,) containing the number of
                likelihood evaluations of each particle.
            steps : numpy.ndarray
                Array of shape (n_particles,) containing the number of steps
                of each particle.
            efficiency : numpy.ndarray
                Array of shape (n_particles,) containing the efficiency of
                each particle.
            ess : numpy.ndarray
                Array of shape (n_particles,) containing the effective sample
                size of each particle.
            accept : numpy.ndarray
                Array of shape (n_particles,) containing the acceptance rate
                of each particle.
            beta : numpy.ndarray
                Array of shape (n_particles,) containing the inverse
                temperature of each particle.
        """
        for key in data.keys():
            if key in self.past.keys():
                value = data.get(key)
                # Save to past states
                self.past.get(key).append(value)

    def pop(self, key):
        """
        Remove the last element of the given key.

        Parameters
        ----------
        key : str
            Key of the element to be removed.
        
        Notes
        -----
        This method is useful to remove the last element of the particles
        after the resampling step.
        """
        _ = self.past.get(key).pop()

    def get(self, key, index=None, flat=False):
        """
        Get the element of the given key.

        Parameters
        ----------
        key : str
            Key of the element to be returned.
        index : int, optional
            Index of the element to be returned. If None, all elements are
            returned.
        flat : bool, optional
            If True, the elements are returned as a flattened array. Otherwise,
            the elements are returned as a numpy.ndarray.
        
        Returns
        -------
        element : numpy.ndarray
            Array of shape (n_particles,) or (n_particles, n_dim) containing
            the elements of the given key.
        
        Notes
        -----
        If index is None, the elements are returned as a numpy.ndarray. If
        index is not None, the elements are returned as a numpy.ndarray with
        shape (n_dim,). If flat is True, the elements are returned as a
        flattened array.

        Examples
        --------
        >>> particles = Particles(n_particles=10, n_dim=2)
        >>> particles.update(dict(u=np.random.randn(10,2)))
        >>> particles.get("u").shape
        (10, 2)
        >>> particles.get("u", index=0).shape
        (2,)
        >>> particles.get("u", index=0, flat=True).shape
        (2,)
        >>> particles.get("u", index=None, flat=True).shape
        (20,)
        """
        if index is None:
            if flat:
                return np.concatenate(self.past.get(key))
            else:
                return np.asarray(self.past.get(key))
        else:
            return self.past.get(key)[index]
        
    def compute_logw_and_logz(self, beta_final=1.0, normalize=True):

        logz = self.get("logz")
        logl = self.get("logl")
        beta = self.get("beta")

        A = logl * beta_final
        b = np.array([logl * beta[i] - logz[i] for i in range(len(beta))])
        B = np.logaddexp.reduce(b, axis=0) - np.log(len(beta))
        logw = A - B
        logw = np.concatenate(logw)
        logz_new = np.logaddexp.reduce(logw) - np.log(len(logw))

        if normalize:
            logw -= np.logaddexp.reduce(logw)

        return logw, logz_new
    
    def compute_results(self):
        """
        Compute the results of the particles.

        Returns
        -------
        results_dict : dict
            Dictionary containing the results of the particles.

        Notes
        -----
        The dictionary contains the following keys:
            u : numpy.ndarray
                Array of shape (n_particles, n_dim) containing the particles.
            logdetj : numpy.ndarray
                Array of shape (n_particles,) containing the log-determinant
                of the Jacobian of the transformation from the unit hypercube
                to the parameter space.
            logl : numpy.ndarray
                Array of shape (n_particles,) containing the log-likelihoods.
            logp : numpy.ndarray
                Array of shape (n_particles,) containing the log-priors.
            logw : numpy.ndarray
                Array of shape (n_particles,) containing the log-weights.
            blobs : numpy.ndarray
                Array of shape (n_particles,) containing the blobs (derived parameters).
            iter : numpy.ndarray
                Array of shape (n_particles,) containing the iteration number
                of each particle.
            logz : numpy.ndarray
                Array of shape (n_particles,) containing the log-evidence of
                each particle.
            calls : numpy.ndarray
                Array of shape (n_particles,) containing the number of
                likelihood evaluations of each particle.
            steps : numpy.ndarray
                Array of shape (n_particles,) containing the number of steps
                of each particle.
            efficiency : numpy.ndarray
                Array of shape (n_particles,) containing the efficiency of
                each particle.
            ess : numpy.ndarray
                Array of shape (n_particles,) containing the effective sample
                size of each particle.
            accept : numpy.ndarray
                Array of shape (n_particles,) containing the acceptance rate
                of each particle.
            beta : numpy.ndarray
                Array of shape (n_particles,) containing the inverse
                temperature of each particle.

        Examples
        --------
        >>> particles = Particles(n_particles=10, n_dim=2)
        >>> particles.update(dict(u=np.random.randn(10,2)))
        >>> particles.compute_results().keys()
        dict_keys(['u', 'logdetj', 'logl', 'logp', 'logw', 'blobs', 'iter', 'logz', 'calls', 'steps', 'efficiency', 'ess', 'accept', 'beta'])
        """
        if self.results_dict is None:
            self.results_dict = dict()
            for key in self.past.keys():
                self.results_dict[key] = self.get(key)

            logw, _ = self.compute_logw_and_logz(1.0)

            self.results_dict["logw"] = logw
            #self.results_dict["ess"] = np.exp(log_ess)

        return self.results_dict

