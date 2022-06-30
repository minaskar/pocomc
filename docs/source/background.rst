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
prior probabilities of the two models. The latter is often set to 1 (i.e. no model is prefered a priori). The 
Bayes factor on the other hand is simply the ratio of the model evidences of the two models, or

.. math::
    BF_{ij} \equiv \frac{p(d\vert\mathcal{M}_{i})}{p(d\vert\mathcal{M}_{j})} = \frac{\mathcal{Z}_{i}}{\mathcal{Z}_{j}}

Preconditioned Monte Carlo
--------------------------

*Preconditioned Monte Carlo (PMC)*, which is the algorithm under the hood of `pocoMC`, address both tasks of
Bayesian *parameter estimation* and *model comparison* in a very efficient and robust way. The basic idea is
to evolve a collection of particles through a sequence of intermediate distributions that connect the prior 
:math:`\pi(\theta)` to the posterior :math:`\mathcal{P}(\theta)`. PMC and `pocoMC` do that by combining the
powerful *Sequential Monte Carlo (SMC)* algorithm with *Normalising Flow Preconditioning*. 

Bridging the prior to the posterior
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
A path of intermediate densities :math:`p_{t}(\theta)` interpolating the prior :math:`\pi(\theta)` and the 
posterior :math:`\mathcal{P}(\theta)` the posterior is defined as

.. math::
    p_{t}(\theta) \equiv \pi (\theta)^{1-\beta_{t}} \mathcal{P}(\theta)^{\beta_{t}} = \pi (\theta) \mathcal{L}(\theta)^{\beta_{t}}

where

.. math::
    0 = \beta_{1} < \beta_{2} < \dots < \beta_{T} = 1

such that :math:`p_{1}=\pi(\theta)` is the prior and :math:`p_{T}=\mathcal{P}(\theta)` is the posterior. The 
number and values of the :math:`beta` levels are determined adaptively during the run based on the *effective sample size (ESS)*
of the population of particles.

Sequential Monte Carlo
^^^^^^^^^^^^^^^^^^^^^^
Initially (i.e. for :math:`t=1`), a collection of :math:`N` particles are drawn from the prior :math:`\theta_{i}\sim\pi(\theta)`
and an initial *importance weight* :math:`W_{i}=1/N` is associated with each particle.

Then, the following three steps are repeated until :math:`t\rightarrow T` and :math:`\beta_{t}\rightarrow 1`:

1. **Correction** - The particles are *reweighted* and their importance weights are updated :math:`W_{i}^{(t)}\leftarrow W_{i}^{(t-1)}\times p_{t}(\theta_{t-1})/p_{t-1}(\theta_{t-1})``.
2. **Selection** - The particles are *resampled* according to their weights :math:`W_{i}`.
3. **Mutation** - Finally, the particles are updated using a number of *Markov chain Monte Carlo (MCMC)* steps.

Once the run is done, the particles are distributed according to the posterior.

An estimate of the *effective sample size (ESS)* of the population of particles can be provided during the run as:

.. math::
    \hat{ESS} = \left( \sum_{i=1}^{N}W_{i}^{2}\right)^{-1}


Normalising Flow Preconditioning
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The purpose of the *mutation* step in SMC is to diversify the particles and allow their distribution to reach the 
equilibrium distribution :math:`p_{t}(\theta)`. To this end, common MCMC methods (e.g. Metropolis-Hastings) are
often employed in practice. However, the sampling efficiency of MCMC is affected by the presense of correlation
between the parameters and multimodality. In order to maintain their efficiency, PMC and `pocoMC` utilise a 
*Normalising Flow (NF)* (i.e. an invertible transformation parameterised by a neural network) to precondition or
decorrelate the parameters of the target distribution. *Normalising Flow Preconditioning* works by approximately
mapping the (potentially correlated) target distribution :math:`p_{t}(\theta)` into a zero-mean unit-variance
normal distribution :math:`\mathcal{N}(0,1)` using the NF transform :math:`u = f(\theta)`. Sampling then proceeds 
in the uncorrelated latent space of :math:`u` and at the end of each iteration the samples are transformed back
to the original space using the inverse transform :math:`\theta = f^{-1}(u)`. Running MCMC in the latent space 
can be orders of magnitude more efficient than in the original parameter space, as the parameters are approximately
uncorrelated and target distribution unimodal.

Hyperparameters
^^^^^^^^^^^^^^^

Finally, PMC and `pocoMC` rely on two sets of hyperparameters to work. The first consists of the normalising flow
hyperparameters (e.g. number of layers, number of neurons, etc.) which the user is advised to leave to their default
values unless they are confident that they know what they are doing. The second set of hyperparameters has to do
with sampling procedure of PMC, and consists of:

1. The number of particles that will be used to sample. The default value is 1000 but more might be needed for challenging or high dimensional target distributions.
2. The *effective sample size (ESS)* that will be maintain constant during the run. The default value is :math:`ESS=0.95`. This effectively determines the schedule of the :math:`\beta_{t}` values as well, with higher ESS resulting in great number of :math:`\beta_{t}` levels and more conservative exploration.
3. The *correlation coefficient (CC)* which takes values in the range :math:`(0,1)` determines how long the particles are propaged with MCMC in each iteration. As the value of CC expresses the mean correlation between the current and initial distribution of particles,  a lower value results in more MCMC steps per iteration and thus more careful and conservative exploration of the target distribution. The default value is :math:`CC=0.75`.