Prior Probability
=================

Standard priors
---------------

The next step is to define the *prior* probability distribution. This encodes our knowledge about the parameters of the model
before we have seen any data.

``pocoMC`` offers two ways to define a prior. The first is to utilise ready-made priors from the ``scipy.stats`` package. For instance,
if we want our prior to be a *uniform* distribution on the interval :math:`[-10,10]` for all 10 of the parameters, we would do::

    from scipy.stats import uniform

    prior = pc.Prior(n_dim * [uniform(loc=-10.0, scale=20.0)]) # Uniform prior on [-10,10] for all 10 parameters.

Suppose now that we want a different prior for each parameter. For instance, we want the first five parameters to have a flat/uniform
prior :math:`x_{i}\sim\mathcal{U}(-10,10)` for :math:`i=0,1,\dots,4` and the last five to have a Gaussian/normal prior
with mean :math:`\mu=0` and standard deviation :math:`\sigma=3`, i.e. :math:`x_{i}\sim\mathcal{N}(0,3^{2})` for :math:`i=5,6,\dots,9`.
We would do::

    from scipy.stats import uniform, norm

    prior = pc.Prior([uniform(loc=-10.0, scale=20.0), # Uniform prior on [-10,10] for the first parameter.
                      uniform(loc=-10.0, scale=20.0), # Uniform prior on [-10,10] for the second parameter.
                      uniform(loc=-10.0, scale=20.0), # Uniform prior on [-10,10] for the third parameter.
                      uniform(loc=-10.0, scale=20.0), # Uniform prior on [-10,10] for the fourth parameter.
                      uniform(loc=-10.0, scale=20.0), # Uniform prior on [-10,10] for the fifth parameter.
                      norm(loc=0.0, scale=3.0), # Normal prior with mean=0 and std=3 for the sixth parameter.
                      norm(loc=0.0, scale=3.0), # Normal prior with mean=0 and std=3 for the seventh parameter.
                      norm(loc=0.0, scale=3.0), # Normal prior with mean=0 and std=3 for the eighth parameter.
                      norm(loc=0.0, scale=3.0), # Normal prior with mean=0 and std=3 for the ninth parameter.
                      norm(loc=0.0, scale=3.0), # Normal prior with mean=0 and std=3 for the tenth parameter.
                     ])

or simply::

    from scipy.stats import uniform, norm

    prior = pc.Prior([uniform(loc=-10.0, scale=20.0)] * 5 + [norm(loc=0.0, scale=3.0)] * 5)

One is free to use any of the priors available in the ``scipy.stats`` package. For a full list see `here <https://docs.scipy.org/doc/scipy/reference/stats.html>`_.

Custom priors
-------------

The second way to define a prior is to define a class including the ``logpdf`` and ``rvs`` methods and ``dim`` 
and ``bounds`` attributes. This can be useful when the prior has some conditional/hierarchical structure.
As an example, let us assume we have a three-parameter model where the prior for the third parameter depends 
on the values for the first two. This might be the case in, e.g., a hierarchical model where the prior over ``c`` 
is a Normal distribution whose mean ``m`` and standard deviation ``s`` are determined by a corresponding 
“hyper-prior”. We can easily set up a prior transform for this model by just going through the variables in order. 
This would look like::
    
        import numpy as np
        from scipy.stats import norm
    
        class CustomPrior:
            def __init__(self):
                self.dim = 3
                self.bounds = np.array([[-np.inf, np.inf], 
                                        [0.0, 10], 
                                        [-np.inf, np.inf]])
                self.hyper_mean = 0.0
                self.hyper_std = 3.0
    
            def logpdf(self, x):
                m, s, c = x
                return norm.logpdf(c, loc=m, scale=s)
    
            def rvs(self, size=1):
                m = np.random.normal(loc=self.hyper_mean, scale=self.hyper_std, size=size)
                s = np.random.uniform(low=0.0, high=10.0, size=size)
                c = np.random.normal(loc=m, scale=s, size=size)
                return np.array([m, s, c]).T

        prior = CustomPrior()


Boundary conditions
-------------------

By default, ``pocoMC`` assumes that all parameters specified in the prior have hard bounds. In other words, each
parameter is free to vary in a prespecified range. Anytime a value is proposed by ``pocoMC`` that lies outside of
this range, it is automatically rejected. This is the desired behavior for most problems, since individual parameters 
are often either defined everywhere (i.e. from negative infinity to infinity) or over a finite range (e.g., from -1 to 
+1).

However, there are problems in which specific parameters may behave differently. ``pocoMC`` supports two such cases:

- **Periodic boundary conditions**. In this case, ``pocoMC`` assumes that the parameter is periodic. For example, 
  if the parameter is on the interval ``[0, 2*np.pi]``, then the parameter can be wrapped around to the other side
  of the interval. This can be useful for phase parameters that might be periodic e.g. on a range ``[0,2*np.pi]``.
- **Reflective boundary conditions**. In this case, ``pocoMC`` assumes that the parameter is reflective. For example,
  if the parameter is on the interval ``[0, 1]``, then the parameter can be flipped around to the other side of the
  interval. This can be useful for parameters that are ratios where ``a/b`` and  ``b/a`` are equivalent.

Given the above, it is possible to set the ``periodic`` and ``reflective`` attributes of the prior. For example, in 
a five-parameter model, if we want the first two parameters to be periodic, and the third and fourth to be reflective, 
we would do::

    from scipy.stats import uniform, norm

    prior = pc.Prior([
        uniform(loc=0.0, scale=2*np.pi), # this parameter is periodic
        uniform(loc=0.0, scale=2*np.pi), # this parameter is periodic
        uniform(loc=0.0, scale=1.0), # this parameter is reflective
        uniform(loc=0.0, scale=1.0), # this parameter is reflective
        norm(loc=0.0, scale=3.0), # this parameter is neither periodic nor reflective
    ])

    sampler = pc.Sampler(prior, 
                        loglike, 
                        periodic=[0,1], 
                        reflective=[2,3])

As you can see, nothing changes in the definition of the prior. Instead, we just need to provide the indices of the
parameters that should be periodic and reflective to the sampler.