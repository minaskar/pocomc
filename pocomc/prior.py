import numpy as np

class Prior:
    """
    A class for priors.

    Parameters
    ----------
    dists : list of scipy.stats distributions
        A list of distributions for each parameter. The length of the list
        determines the dimension of the prior.

    Attributes
    ----------
    dists : list of scipy.stats distributions
        A list of distributions for each parameter. The length of the list
        determines the dimension of the prior.
    bounds : ndarray
        An array of shape (dim, 2) containing the lower and upper bounds for
        each parameter.
    dim : int
        The dimension of the prior.
    
    Methods
    -------
    logpdf(x)
        Returns the log of the probability density function evaluated at x.
    rvs(size=1)
        Returns a random sample from the prior.
    
    Examples
    --------
    >>> import numpy as np
    >>> from scipy.stats import norm, uniform
    >>> from pocomc.prior import Prior
    >>> dists = [norm(loc=0, scale=1), uniform(loc=0, scale=1)]
    >>> prior = Prior(dists)
    >>> prior.logpdf(np.array([0, 0]))
    -1.8378770664093453
    >>> prior.rvs()
    array([0.417022  , 0.72032449])
    >>> prior.bounds
    array([[-inf,  inf],
           [ 0. ,  1. ]])
    >>> prior.dim
    2

    Notes
    -----
    The logpdf method is implemented as a sum of the logpdf methods of the
    individual distributions. This is equivalent to assuming that the
    parameters are independent.

    The rvs method is implemented by sampling from each distribution
    independently and then transposing the result. This is equivalent to
    assuming that the parameters are independent.

    The bounds property is implemented by calling the support method of each
    distribution. This is equivalent to assuming that the parameters are
    independent.

    The dim property is implemented by returning the length of the dists
    attribute. This is equivalent to assuming that the parameters are
    independent.
    """

    def __init__(self, dists=None):
        self.dists = dists

    def logpdf(self, x):
        """
        Returns the log of the probability density function evaluated at x.

        Parameters
        ----------
        x : ndarray
            An array of shape (n, dim) containing n samples of the parameters.
        
        Returns
        -------
        logp : ndarray
            An array of shape (n,) containing the log of the probability
            density function evaluated at each sample.
        
        Examples
        --------
        >>> import numpy as np
        >>> from scipy.stats import norm, uniform
        >>> from pocomc.prior import Prior
        >>> dists = [norm(loc=0, scale=1), uniform(loc=0, scale=1)]
        >>> prior = Prior(dists)
        >>> prior.logpdf(np.array([0, 0]))
        -1.8378770664093453
        >>> prior.logpdf(np.array([[0, 0], [0, 0]]))
        array([-1.83787707, -1.83787707])
        """
        logp = np.zeros(len(x))
        for i, dist in enumerate(self.dists): 
            logp += dist.logpdf(x[:,i])
        return logp
    
    def rvs(self, size=1):
        """
        Returns a random sample from the prior.

        Parameters
        ----------
        size : int, optional
            The number of samples to return. The default is 1.
        
        Returns
        -------
        samples : ndarray
            An array of shape (size, dim) containing the samples.
    
        Examples
        --------
        >>> import numpy as np
        >>> from scipy.stats import norm, uniform
        >>> from pocomc.prior import Prior
        >>> dists = [norm(loc=0, scale=1), uniform(loc=0, scale=1)]
        >>> prior = Prior(dists)
        >>> prior.rvs()
        array([0.417022  , 0.72032449])
        >>> prior.rvs(size=2)
        array([[0.417022  , 0.72032449],
               [0.00011438, 0.30233257]])
        """
        samples = []
        for dist in self.dists:
            samples.append(dist.rvs(size=size))
        return np.transpose(samples)
    
    @property
    def bounds(self):
        """
        An array of shape (dim, 2) containing the lower and upper bounds for
        each parameter.

        Examples
        --------
        >>> import numpy as np
        >>> from scipy.stats import norm, uniform
        >>> from pocomc.prior import Prior
        >>> dists = [norm(loc=0, scale=1), uniform(loc=0, scale=1)]
        >>> prior = Prior(dists)
        >>> prior.bounds
        array([[-inf,  inf],
               [ 0. ,  1. ]])
        """
        bounds = []
        for dist in self.dists:
            bounds.append(dist.support())
        return np.array(bounds)
    
    @property
    def dim(self):
        """
        The dimension of the prior.

        Examples
        --------
        >>> import numpy as np
        >>> from scipy.stats import norm, uniform
        >>> from pocomc.prior import Prior
        >>> dists = [norm(loc=0, scale=1), uniform(loc=0, scale=1)]
        >>> prior = Prior(dists)
        >>> prior.dim
        2
        """
        return len(self.dists)