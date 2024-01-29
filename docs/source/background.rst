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

