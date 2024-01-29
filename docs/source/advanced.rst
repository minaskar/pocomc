.. _advanced:

==============
Advanced Guide
==============

This guide is intended to give the user an idea of the various options and possibilities available in ``pocoMC``. 
We will start by explaining in detail how to define a problem of Bayesian inference. Then, we will demonstrate 
how to use ``pocoMC`` in order to solve the aforementioned problem as effectively and robustly as possible.

Likelihood function
===================

We will begin by defining our problem. To this end we need to define *log-likelihood* function :math:`\log\mathcal{L}(\theta)=\log p(d\vert\theta,\mathcal{M})` and 
*log-prior* probability density function :math:`\log \pi(\theta) = \log p(\theta\vert \mathcal{M})`. We will start with the former, the likelihood.
If you are not familiar with these terms I encourage you to visit the :doc:`background` section for some theory. 

Suppose that we want our *likelihood* function to be a *Gaussian density* with 10 parameters or in 10-D, we would do
something like::

    import numpy as np

    # Define the dimensionality of our problem.
    n_dim = 10

    # Define our 3-D correlated multivariate normal log-likelihood.
    C = np.identity(n_dim)
    C[C==0] = 0.95
    Cinv = np.linalg.inv(C)
    lnorm = -0.5 * (np.log(2 * np.pi) * n_dim + np.log(np.linalg.det(C)))

    def log_like(x):
        return -0.5 * np.dot(x, np.dot(Cinv, x)) + lnorm

The inclusion of the normalisation factor ``lnorm`` is not strictly necessary as it does not depend on ``x`` and thus does 
vary. 


Prior probability distribution
==============================

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


Sampler options
===============

Having defined the Bayesian components of the problem (e.g. likelihood, prior, etc.) we can now turn our attention to
configuring ``pocoMC`` in order to solve this inference problem.

The next step is to import ``pocoMC`` and initialise the ``Sampler`` class::

    import pocomc as pc

    sampler = pc.Sampler(n_particles = n_particles,
                         n_dim = n_dim,
                         log_likelihood = log_like,
                         log_prior = log_prior,
                         bounds = bounds,
                        )

The sampler also accepts other arguments, for a full list see :doc:`api`. Those include:
 
- Additional arguments passed to the log-likelihood using the arguments ``log_likelihood_args`` and ``log_likelihood_kwargs``,
  or to the log-prior using the arguments ``log_prior_args`` and ``log_prior_kwargs``.
- The arguments ``vectorize_likelihood`` and ``vectorize_prior`` which accept boolean values allow the user to use vectorized
  log-likelihood and log-prior functions.
- The ``periodic`` and  ``reflective`` arguments that accept a list of indices corresponding to parameters of the model that
  have *periodic* or *reflective* boundary conditions. The first kind include *phase* parameters that might be periodic e.g. 
  on a range :math:`[0,2\pi]`. The latter can arise in cases where parameters are ratios where :math:`a/b` and :math:`b/a`
  are equivalent.
- The Sampler class also accepts a dictionary ``flow_config`` with various options for the configuration of the normalising
  flow. An example showing some of the default values and what each parameter means is shown below::

    flow_config = dict(n_blocks = 6, # Number of blocks
                       hidden_size = 3 * ndim, # Number of neurons per layer
                       n_hidden = 1, # Number of layers per block
                       flow_type = 'maf' # Type of normalising flow. Options include 'maf' and 'realnvp'
                      )

- Apart from the ``flow_config``, the sampler accepts the ``train_config`` dictionary which includes arguments related to
  the training procedure of the normalising flow. An example showing some of the default values and what each parameter
  means is shown below::

    train_config = dict(validation_split = 0.2, # Percentage of particles to use for validation
                        epochs = 1000, # Maximum number of epochs
                        batch_size = n_particles, # Batch size used for training
                        patience = 30, # Number of iterations to wait with no improvement in the (monitor) loss until stopping.
                        monitor = 'val_loss', # Which loss to monitor for early stopping. Options are 'val_loss' and 'loss'.
                        shuffle = True, # Shuffle the particles
                        lr = [1e-2, 1e-3, 1e-4, 1e-5], # Learning rates. If more than one is provided then they are used as an annealing schedule.
                        weight_decay = 1e-8, # Weight decay parameter.
                        clip_grad_norm = 1.0, # Clip huge gradients to avoid training issues.
                        laplace_prior_scale = 0.2, # Scale of the Laplace prior put on the weights.
                        gaussian_prior_scale = None, # Scale of the Gaussian prior put on the weights.
                        device = 'cpu', # Device to use for training. Currently only 'cpu' is supported.
                       )
  

Running the sampler
-------------------

Running the actual sampling procedure that will produce, among other things, a collection of samples from the posterior as well as 
an unbiased estimate of the model evidence, can be done by providing the ``prior_samples`` to the ``run`` method of the sampler::

    sampler.run(prior_samples)

Running the above also produces a progress bar similar to the one shown below::

    Iter: 6it [00:17,  3.18s/it, beta=0.00239, calls=35000, ESS=0.95, logZ=-3.52, accept=0.232, N=6, scale=0.964, corr=0.728] 

The ``scale`` parameter in the above progress bar is perhaps the most important metric of the sampling performance. It is defined
as the ratio of the actual Metropolis-Hastings proposal scale (in latent space) to the optimal one :math:`2.38/\sqrt{D}`. Its 
value reflects the quality of the NF preconditioner. A value of ``scale=1.0`` corresponds to perfect preconditioning and maximum
sampling efficiency. It is normal for this value to drop up to ``0.5`` during the run when the NF struggles to capture the
geometry of the posterior. However, if the value of the ``scale`` parameter becomes significantly less than one (e.g. ``0.1``)
this is usually an indication that something is wrong. A possible way to increase the scale parameter is to increase the number
of particles.

We can also use the ``run`` method to specify the desired *effective sample size (ESS)*, the :math:`\gamma` factor, as well as
the minimum and maximum number of MCMC steps per iteration (the actual number is determined adaptively)::

    sampler.run(prior_samples = prior_samples,
                ess = 0.95,
                gamma = 0.75,
                nmin = 5,
                nmax = 50
               )


Results
-------

Once the run is complete and we have optionally added extra samples, it is time to look at the results. This can be done using the 
``results`` dictionary, as follows::

    results = sampler.results

This is a dictionary which includes the following arrays:

1. ``results['samples']`` - Array with the **samples drawn from posterior**. This is usually what you need for parameter inference.
2. ``results['loglikelihood']`` - Array with the **values of the log-likelihood** for the posterior samples given by ``results['samples']``.
3. ``results['logprior']`` - Array with the **values of the log-prior** for the posterior samples given by ``results['samples']``.
4. ``results['logz']`` - Array with the evolution of the estimate of the **logarithm of the model evidence** :math:`\log\mathcal{Z}`. This is usually what you need for model comparison.
5. ``results['iter']`` - Array with number iteration indices (e.g. ``np.array([0, 1, 2, ...])``)
6. ``results['x']`` - Array with the final samples from all the intermediate distributions.
7. ``results['logl']`` - Array with the values of the log-likelihood for the samples from all the intermediate distributions.
8. ``results['logp']`` - Array with the values of the log-prior for the samples from all the intermediate distributions.
9.  ``results['logw']`` - Array with the values of the log-weights for the samples from all the intermediate distributions.
10. ``results['ess']`` - Array with the evolution of the ESS during the run.
11. ``results['ncall']`` - Array with the evolution of the number of log-likelihood calls during the run.
12. ``results['beta']`` - Array with the values of beta.
13. ``results['accept']`` - Array with the Metropolis-Hastings acceptance rates during the run.
14. ``results['scale']`` - Array with the evolution of the scale factor during the run.
15. ``results['steps']`` - Array with the number of MCMC steps per iteration during the run.


Parallelisation
===============

If you want to run computations in parallel, ``pocoMC`` can use a user-defined ``pool`` to execute a variety of expensive operations 
in parallel rather than in serial. This can be done by passing the ``pool`` object to the sampler upon initialization::

    sampler = pc.Sampler(prior=prior,
                         likelihood = log_like,
                         pool = pool,
                        )
    sampler.run()

By default ``pocoMC`` will use the ``pool`` to execute the calculation of the ``log_likelihood`` in parallel for the particles.

Commonly used pools are offered by standard Python in the ``multiprocessing`` package and the ``multiprocess`` package. The benefit of
the latter is that it uses ``dill`` to perform the serialization so it can actually work with a greater variety of log-likelihood
functions. The disadvantage is that it needs to be installed manually. An example of how to use such a pool is the following::

    from multiprocessing import Pool 

    n_cpus = 4

    with Pool(n_cpus) as pool:

        sampler = pc.Sampler(prior=prior,
                             likelihood = log_like,
                             pool = pool,
                            )
        
        sampler.run()

where ``n_cpus`` is the number of available CPUs in our machine. Since ``numpy`` and ``torch`` are doing some internal parallelisation
it is a good idea to specify how many CPUs should be used for that using::

    import os

    os.environ["OMP_NUM_THREADS"] = "1"

at the beginning of the code. This can affect the speed of the normalising flow training.

Finally, other pools can also be used, particularly if you plan to use ``pocoMC`` is a supercomputing cluster you may want to use
an ``mpi4py`` pool so that you can utilise multiple nodes.

The speed-up offered by parallelisation in ``pocoMC`` is expected to be linear in the number of particles.


Saving and resuming runs
========================

A useful option, especially for long runs, is to be able to store the state of ``pocoMC`` in a file and also the to use
that file in order to later continue the same run. This can help avoid disastrous situations in which a run is interrupted
or terminated prematurely (e.g. due to time limitation in computing clusters or possible crashes).

Fortunately, ``pocoMC`` offers both options to save and load a previous state of the sampler.

Saving the state of the sampler
-------------------------------

In order to save the state of the sampler during the run, one has to specify how often to save the state in a file. This is
done using the ``save_every`` argument in the ``run`` method. The default is ``save_every=None`` which means that no state
is saved during the run. If instead we want to store the state of ``pocoMC`` every e.g. ``3`` iterations, we would do
something like::

    sampler.run(
        save_every = 3,
    )

The default directory in which the state files are saved is a folder named ``states`` in the current directory. One can change
this using the ``output_dir`` argument when initialising the sampler (e.g. ``output_dir = "new_run"``). By default, the state
files follow the naming convention ``pmc_{i}.state`` where ``i`` is the iteration index. For instance, if ``save_every=3`` was 
specified then the ``output_dir`` directory will include the files ``pmc_3.state``, ``pmc_6.state``, etc. One can also change
the label from ``pmc`` to anything else by using the ``output_label`` argument when initialising the sampler (e.g. 
``output_label="grav_waves"``).

Loading the state of the sampler
--------------------------------

Loading a previous state of the sampler and resuming the run from that point requires to provide the path to the specific state
file to the ``run`` method using the ``resume_state_path`` argument. For instance, if we want to continue the run from the 
``pmc_3.state`` which is in the ``states`` directory, we would do::

    sampler.run(
        resume_state_path = "states/pmc_3.state"
    )
