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

    sampler = pc.Sampler(prior = prior,
                         likelihood = log_like,
                        )

The sampler also accepts other arguments, for a full list see :doc:`api`. Those include:

- An optional ``n_dim`` argument that specifies the dimensionality of the problem. If not provided, the sampler will
  try to infer it from the ``prior``.

- An optional ``n_ess`` argument that specifies the number of effectively independent particles to use. If not provided, 
  ``1000`` is used by default. This determines the speed and robustness of the sampling procedure. Generally, higher values
  of ``n_ess`` are better but they also require more computational resources. Higher-dimensional problems or problems with
  complex posteriors may require higher values of ``n_ess``.

- An optional ``n_active`` argument that specifies the number of active particles to use. If not provided, ``250`` is used
  by default. Generally, ``n_active`` should be less than ``n_ess // 2``. The active particles are the ones (out of the total 
  ``n_ess`` particles) that are updated in each iteration. Therefore, one can think of ``n_active`` as the batch size used
  for the sampling procedure. 

- Additional arguments passed to the log-likelihood using the arguments ``likelihood_args`` and ``likelihood_kwargs``. For instance,
  if the log-likelihood function accepts an extra argument ``data`` we would do something like::

    sampler = pc.Sampler(prior = prior,
                         likelihood = log_like,
                         likelihood_args = [data],
                        )

- The argument ``vectorize`` that accepts a boolean value (i.e., ``True`` or ``False``) allows the user to use a vectorized
  log-likelihood function. This can be useful when the vectorized log-likelihood function is cheaper to evaluate or a parallel
  version of the log-likelihood function is available. For instance, if we have a vectorized log-likelihood function ``log_like_vec``
  we would do something like::
    
        sampler = pc.Sampler(prior = prior,
                             likelihood = log_like_vec,
                             vectorize = True,
                            )

- The ``pool`` argument allows the user to specify a ``pool`` object to use for parallelisation. For more details see the
  Parallelisation section below.

- The ``flow`` argument allows the user to specify the normalising flow to use. For more details see the Normalizing Flow section below
  section. Generally, the default normalising flow is a good choice for most problems.

- The ``train_config`` argument allows the user to specify the training configuration of the normalising flow. For more details
  see the Normalizing Flow section below. Generally, the default training configuration is a good choice for most problems.

- The ``precondition`` argument allows the user to specify whether to use the preconditioning of the normalising flow. For more
  details see the Normalizing Flow section below. Generally, using normalizing flow preconditioning is a good choice for most
  problems as the latent space is usually less correlated than the parameter space.

- The ``n_prior`` argument allows the user to specify the number of prior samples to draw in the beginning of the run. By default
  ``n_prior=2*(n_ess//n_active)*n_active`` is selected. This is useful for the initialisation of the sampler and we do not recommend changing it.

- The ``sample`` argument determines the MCMC sampling method to use. By default ``sampler='pcn'`` is selected and the preconditioned
  Crank-Nicolson (PCN) method is used. The alternative is to use ``sampler='rwm'`` and the standard Random-walk Metropolis method.

- The ``max_steps`` argument determines the maximum number of MCMC steps to use per iteration. By default ``max_steps=5*n_dim`` is selected.
  This is the maximum number of steps that the MCMC sampler will take in order to update the active particles. The actual number of
  steps is determined adaptively by the sampler. The default value of ``max_steps`` is usually a good choice for most problems.

- The ``patience`` arguments determines the maximum number of MCMC steps after which the sampler will stop updating the active particles
  if the average ``logP`` has not increased. By default this parameter is determined automatically by the sampler and depends on the dimensionality
  of the problem, the acceptance rate, and the proposal scale. The default value of ``patience`` is usually a good choice for most problems
  and we do not recommend changing it.

- The ``ess_threshold`` defines the minimum effective sample size (ESS) of an iteration for the sampler to consider using particles from
  that iteration to update the active particles. By default ``ess_threshold=4*n_dim`` is selected. This means that if the ESS of an iteration
  is less than ``4*n_dim`` then the sampler will not use any particles from that iteration to update the active particles during resampling.
  The default value of ``ess_threshold`` is usually a good choice for most problems and we do not recommend changing it.

- The ``output_dir`` argument allows the user to specify the directory in which the output files will be saved. By default ``output_dir=None``
  is selected and the output files will be saved in the ``state`` directory.

- The ``output_label`` is the label used in state files. Defaullt is ``None`` which corresponds to ``"pmc"``. The saved states are named
  as ``"{output_dir}/{output_label}_{i}.state"`` where ``i`` is the iteration index.  Output files can be used to resume a run.

- The ``random_state`` argument allows the user to specify the random state of the sampler (i.e., any integer value). By default
  ``random_state=None`` is selected and the random state is not fixed. This can be useful for debugging purposes.

Running the sampler
-------------------

Running the actual sampling procedure that will produce, among other things, a collection of samples from the posterior as well as 
an unbiased estimate of the model evidence, can be done by using the ``run`` method of the sampler::

    sampler.run()

Running the above also produces a progress bar similar to the one shown below::

    Iter: 64it [03:50,  3.59s/it, calls=77500, beta=1, logZ=-21.5, ESS=5e+3, acc=0.704, steps=3, logP=-25.2, eff=1]

The ``calls`` argument shows the total number of log-likelihood calls. The ``beta`` argument shows the current value of the inverse
temperature. The ``logZ`` argument shows the current estimate of the logarithm of the model evidence. The ``ESS`` argument shows the
current estimate of the effective sample size. The ``acc`` argument shows the current acceptance rate. The ``steps`` argument shows
the current number of MCMC steps per iteration. The ``logP`` argument shows the current average log-posterior. The ``eff`` argument
shows the current efficiency of the sampling procedure. Generally, as long as the acceptance rate remains above ``0.15`` and the
efficiency remains above ``0.2`` the sampling procedure is working well. Low values of the efficiency can indicate that the normalizing
flow is struggling to approximate the posterior. In such cases, one can try to use a more powerful normalizing flow (e.g. more layers
or more hidden units per layer) or a different training configuration (e.g. more epochs or a smaller learning rate).


We can also use the ``run`` method to specify the desired number of total effective samples and the number of evidence samples to use
for the estimate of the model evidence. For instance, if we want to use ``5000`` total effective samples and ``5000`` evidence samples
we would do something like::

    sampler.run(n_total=5000,
                n_evidence=5000,
                progress=True,
               )

The ``progress`` argument allows the user to specify whether to show a progress bar or not. By default ``progress=True`` is selected
and a progress bar is shown. If ``progress=False`` is selected then no progress bar is shown. This can be useful when running the
sampler in a computing cluster.


Results
-------

Once the run is complete we can look at the results. This can be done in two ways. The first is to use the ``posterior`` and ``evidence``
methods of the sampler. For instance, if we want to get the samples from the posterior we would do::

    samples, weights, logl, logp = sampler.posterior()

The ``samples`` argument is an array with the samples from the posterior. The ``weights`` argument is an array with the weights of the
samples from the posterior. The ``logl`` argument is an array with the values of the log-likelihood for the samples from the posterior.
The ``logp`` argument is an array with the values of the log-prior for the samples from the posterior.

If we want to get samples from the posterior without the weights we would do::

    samples, logl, logp = sampler.posterior(resample=True)

This resamples the particles and is useful when we want to use the samples for parameter inference and we do not want to deal with the weights.

The samples from the posterior can be used for parameter inference. For instance, we can get the mean and standard deviation of the
posterior for each parameter by doing::

    mean = np.mean(samples, axis=0)
    std = np.std(samples, axis=0)

Or, we can utilize a third-party package such as ``corner`` (`available here <https://corner.readthedocs.io/en/latest/>`_) to plot the posterior samples::

    import corner

    fig = corner.corner(samples, labels=[f"$x_{i}$" for i in range(n_dim)]); # If we do not want to use the weights.
    # fig = corner.corner(samples, weights=weights, labels=[f"$x_{i}$" for i in range(n_dim)]); # If we want to use the weights.

Similarly, we can also get the estimate of the model evidence / marginal likelihood by doing::

    logZ, logZerr = sampler.evidence()

The ``logZ`` argument is the estimate of the logarithm of the model evidence. The ``logZerr`` argument is the error on the estimate
of the logarithm of the model evidence. The error is estimated using the bootstrap method.

An alternative, and more advanced way, to look at the results is to use the ``results`` dictionary of the sampler, as follows::

    results = sampler.results

This is a dictionary includes the following keys::

    ``u``, ``x``, ``logdetj``, ``logl``, ``logp``, ``logw``, ``iter``, ``logz``, ``calls``, ``steps``, ``efficiency``, ``ess``, ``accept``, ``beta``.

The ``u`` key is an array with the samples from the latent space. The ``x`` key is an array with the samples from the parameter space.
The ``logdetj`` key is an array with the values of the log-determinant of the Jacobian of the normalizing flow for each sample. The ``logl``
key is an array with the values of the log-likelihood for each sample. The ``logp`` key is an array with the values of the log-prior for
each sample. The ``logw`` key is an array with the values of the log-importance weights for each sample. The ``iter`` key is an array with
the iteration index for each sample. The ``logz`` key is an array with the values of the logarithm of the model evidence for each iteration.
The ``calls`` key is an array with the total number of log-likelihood calls for each iteration. The ``steps`` key is an array with the
number of MCMC steps per iteration. The ``efficiency`` key is an array with the efficiency of the sampling procedure for each iteration.
The ``ess`` key is an array with the effective sample size for each iteration. The ``accept`` key is an array with the acceptance rate
for each iteration. The ``beta`` key is an array with the value of the inverse temperature for each iteration.

Normalizing Flow
================

The default normalizing flow used by ``pocoMC`` is a Masked Autoregressive Flow (MAF) with 6 blocks of 3 layers each. Each layer has 64
hidden units with a residual connection and uses a ``relu`` activation function. Both the normalizing flow and the training configuration
can be changed by the user. For instance, if we want to use a MAF with 12 blocks of 2 layers each and 128 hidden units per layer we would do::

    import zuko

    flow = zuko.flows.MAF(n_dim, 
                          transforms=12, 
                          hidden_features=[128] * 3,
                          residual=True,
                         )

    sampler = pc.Sampler(prior=prior,
                         likelihood = log_like,
                         flow = flow,
                        )

Any normalizing flow provided by the ``zuko`` package can be used. For a full list `see here <https://zuko.readthedocs.io/en/stable/index.html>`_.

Additionally, the training configuration of the normalizing flow can also be changed as follows::

    sampler = pc.Sampler(prior = prior,
                         likelihood = log_like,
                         flow = flow,
                         train_config=dict(validation_split=0.5,
                                           epochs=2000,
                                           batch_size=512,
                                           patience=50,
                                           learning_rate=1e-3,
                                           annealing=False,
                                           gaussian_scale=None,
                                           laplace_scale=None,
                                           noise=None,
                                           shuffle=True,
                                           clip_grad_norm=1.0,
                                           verbose=0,
                                          )
                        )

The above is the default training configuration and is a good choice for most problems. The ``validation_split`` argument determines
the fraction of the training data to use as validation data. The ``epochs`` argument determines the maximum number of epochs to use for training.
The ``batch_size`` argument determines the batch size to use for training. The ``patience`` argument determines the number of epochs to wait
before early stopping if the validation loss does not improve. The ``learning_rate`` argument determines the learning rate to use for training.
The ``annealing`` argument determines whether to use learning rate annealing or not. The ``gaussian_scale`` argument determines the scale of the Gaussian 
prior applied to the weights of the normalizing flow. The ``laplace_scale`` argument determines the scale of the Laplace prior applied to the weights
of the normalizing flow. The ``noise`` argument determines the standard deviation of the Gaussian noise to add to the input of the normalizing flow.
The ``shuffle`` argument determines whether to shuffle the training data or not. The ``clip_grad_norm`` argument determines the maximum norm of the
gradients to use for training. The ``verbose`` argument determines whether to print the training progress or not.                       

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
