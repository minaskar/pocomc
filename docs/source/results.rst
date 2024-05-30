Results
=======

Simple
------

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

Advanced
--------

An alternative, and more advanced way, to look at the results is to use the ``results`` dictionary of the sampler, as follows::

    results = sampler.results

This is a dictionary includes the following keys::

    ``u``, ``x``, ``logdetj``, ``logl``, ``logp``, ``logw``, ``blobs``, ``iter``, ``logz``, ``calls``, ``steps``, ``efficiency``, ``ess``, ``accept``, ``beta``.

The ``u`` key is an array with the samples from the latent space. The ``x`` key is an array with the samples from the parameter space.
The ``logdetj`` key is an array with the values of the log-determinant of the Jacobian of the normalizing flow for each sample. The ``logl``
key is an array with the values of the log-likelihood for each sample. The ``logp`` key is an array with the values of the log-prior for
each sample. The ``logw`` key is an array with the values of the log-importance weights for each sample. The ``blobs`` key is an array with
the values of the blobs for each sample. Blobs are additional quantities that are computed during the sampling procedure. The ``iter`` key is an array with
the iteration index for each sample. The ``logz`` key is an array with the values of the logarithm of the model evidence for each iteration.
The ``calls`` key is an array with the total number of log-likelihood calls for each iteration. The ``steps`` key is an array with the
number of MCMC steps per iteration. The ``efficiency`` key is an array with the efficiency of the sampling procedure for each iteration.
The ``ess`` key is an array with the effective sample size for each iteration. The ``accept`` key is an array with the acceptance rate
for each iteration. The ``beta`` key is an array with the value of the inverse temperature for each iteration.