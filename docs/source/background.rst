.. _background:

Background
==========


Bayesian inference
------------------

In the Bayesian context, one is often interested to approximate the posterior distribution :math:`p(\theta\vert d,\mathcal{M})`,
that is, the probability distribution of the parameters :math:`\theta` given the data :math:`d`
and the model :math:`\mathcal{M}`. This is given by Bayes' theorem:

.. math::
    p(\theta\vert d,\mathcal{M})= \frac{p(d\vert \theta,\mathcal{M})p(\theta\vert\mathcal{M})}{p(d\vert\mathcal{M})}

where


Preconditioned Monte Carlo
--------------------------