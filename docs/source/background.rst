.. _background:

Background
==========


Bayesian inference
------------------

In the Bayesian context, one is often interested to approximate the *posterior distribution* :math:`\mathcal{P}(\theta)\equiv p(\theta\vert d,\mathcal{M})`,
that is, the probability distribution of the parameters :math:`\theta` given the data :math:`d`
and the model :math:`\mathcal{M}`. This is given by Bayes' theorem:

.. math::
    p(\theta\vert d,\mathcal{M})= \frac{p(d\vert \theta,\mathcal{M})p(\theta\vert\mathcal{M})}{p(d\vert\mathcal{M})}

where

.. math::
    \mathcal{L}(\theta) \equiv p(d\vert \theta,\mathcal{M})

is the *likelihood function*,

.. math::
    \pi(\theta) \equiv p(\theta\vert\mathcal{M})

is the *prior probability density*, and

.. math::
    \mathcal{Z} \equiv p(d\vert\mathcal{M})

is the so called *model evidence* or *marginal likelihood*.

Parameter estimation
^^^^^^^^^^^^^^^^^^^^

The task of parameter estimation consists of finding the probability distribution of the parameters :math:`\theta`
of a model :math:`\mathcal{M}` given some data :math:`d`. In practice this is achieved by approximating the 
posterior distribution by a collection of *samples*. The distribution of these samples can then be used to 
approximate various expectation values (e.g. mean, median, standard deviation, credible intervals, 1-D and 
2-D marginal posteriors etc.)

.. math::
    \mathbb{E}_{\mathcal{P}(\theta)}\left[ f(\theta)\right] \equiv \int f(\theta) \mathcal{P}(\theta) d\theta = \sum_{i=1}^{n}f(\theta_{i})

as sums over the samples drawn from the posterior

.. math::
    \theta_{i} \sim \mathcal{P}(\theta)

Model comparison
^^^^^^^^^^^^^^^^

For the task of Bayesian model comparison, one is interested in the ratio of posterior probabilities of models
:math:`\mathcal{M}_{i}` and :math:`\mathcal{M}_{j}`, given by

.. math::
    \frac{p(\mathcal{M}_{i}\vert d)}{p(\mathcal{M}_{j}\vert d)} = \frac{p(d\vert\mathcal{M}_{i})}{p(d\vert\mathcal{M}_{j})} \times \frac{p(\mathcal{M}_{i})}{p(\mathcal{M}_{j})}

where the first term on the right-hand-side is the so called *Bayes factor* and the second term is the ratio of
prior probabilities of the two models. The latter is often set to 1 (i.e. no model is preferred a priori). The
Bayes factor on the other hand is simply the ratio of the model evidences of the two models, or

.. math::
    BF_{ij} \equiv \frac{p(d\vert\mathcal{M}_{i})}{p(d\vert\mathcal{M}_{j})} = \frac{\mathcal{Z}_{i}}{\mathcal{Z}_{j}}


Preconditioned Monte Carlo
--------------------------

The Preconditioned Monte Carlo (PMC) algorithm is a variant of the Persistent Sampling (PS) framework, which is a generalization
of the Sequential Monte Carlo (SMC) algorithm. The PMC algorithm is designed to sample from a sequence of probability distributions
:math:`\mathcal{P}_{t}(\theta)`, where the target distribution :math:`\mathcal{P}_{t}(\theta)` is defined by

.. math::
    \mathcal{P}_{t}(\theta) \propto \mathcal{L}(\theta)^{\beta_{t}}\pi(\theta)

where :math:`\mathcal{L}(\theta)` is the likelihood function and :math:`\pi(\theta)` is the prior probability density. The effective
inverse temperature parameter :math:`\beta_{t}` is initialized to 0 and is gradually increased to 1. When :math:`\beta_{t}=0`, the target
distribution is the prior distribution, and when :math:`\beta_{t}=1`, the target distribution is the posterior distribution. The inverse
temperature parameter is increased in each iteration by a small step size :math:`\Delta\beta` until it reaches 1. The :math:`\Delta\beta`
is computed adaptively in each iteration to ensure PMC maintains a constant number of effective particles. In each iteration, the PMC 
algorithm samples from the target distribution :math:`\mathcal{P}_{t}(\theta)` using a set of particles by applying a sequence of three steps:

1. **Reweighting**: The particles are reweighted to target the distribution :math:`\mathcal{P}_{t}(\theta)`.
2. **Resampling**: The particles are resampled according to their weights to ensure that the effective number of particles is constant.
3. **Mutation**: The particles are mutated by applying a number of MCMC.

The PMC algorithm terminates when the inverse temperature parameter reaches 1. The samples obtained from the PMC algorithm can be used to
approximate the posterior distribution of the parameters :math:`\theta` given the data :math:`d` and the model :math:`\mathcal{M}`. The PMC
algorithm is particularly useful for sampling from high-dimensional and multimodal posterior distributions. Furthemore, the PMC algorithm
offers an estimate of the logarithm of the model evidence :math:`\log\mathcal{Z}` which can be used for Bayesian model comparison.

The high sampling efficiency and robustness of the PMC algorithm is derived by three key features:

1. **Persistent Sampling**: The PMC algorithm maintains a set of particles throughout the entire run of the algorithm. This allows the PMC
   algorithm to reuse the particles from previous iterations to sample from the target distribution in the current iteration. This is particularly
   useful when the target distribution changes smoothly from one iteration to the next.
2. **Normalizing Flow Preconditioning**: The PMC algorithm uses a normalizing flow to precondition each target distribution :math:`\mathcal{P}_{t}(\theta)`.
   The normalizing flow is a sequence of invertible transformations that maps a simple distribution to the target distribution. The normalizing
   flow is trained to approximate the target distribution using a set of particles. Sampling in the target distribution is then performed by
   sampling from the simple distribution and applying the inverse of the normalizing flow. The normalizing flow preconditioning allows the PMC
   algorithm to sample from complex and multimodal target distributions.
3. **t-preconditioned Crank-Nicolson**: The PMC algorithm uses a t-preconditioned Crank-Nicolson integrator to evolve the particles in the target
   distribution. The t-preconditioned Crank-Nicolson algorithm is an MCMC method that scales well with the dimensionality of the target distribution.
   For targets that are close to Gaussian, the t-preconditioned Crank-Nicolson algorithm is particularly efficient and can scale to very high dimensions.
   For non-Gaussian targets (e.g., multimodal distributions), the t-preconditioned Crank-Nicolson algorithm can be combined with the normalizing flow
   preconditioning to sample from the target distribution efficiently even in high dimensions.

Unlike traditional samplers that rely on Random-walk Metropolis, Slice Sampling, Rejection Sampling, Importance Sampling, or Independence Metropolis, PMC
can scale to high-dimensions without desolving into random-walk behavior.