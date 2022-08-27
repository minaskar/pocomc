========
Examples
========

This page highlights several examples on how ``pocomc``
can be used in practice, illustrating both simple and more advanced
aspects of the code. Jupyter notebooks containing more details are available
`on Github <https://github.com/minaskar/pocomc/tree/examples>`_. ``seaborn`` 
is required to be installed to generate some of the plots.

Gaussian Shells
===============

The ``Gaussian Shells`` are a clear example of a bimodal distribution. Although
the example is only in 2D it is a difficult target for many MCMC methods due to 
its peculiar geometry in each mode.

2D marginal posterior
---------------------

.. image:: ./images/gaussian_shells_2d.png
    :align: center

Trace plot
----------

.. image:: ./images/gaussian_shells_trace.png
    :align: center

Run plot
--------

.. image:: ./images/gaussian_shells_run.png
    :align: center

Double Gaussian
===============

The ``Double Gaussian`` distribution consists of a Gaussian mixture in 10D in
which the two components are well separated from each other and one of them is
twice the size of the other.

2D marginal posterior
---------------------

.. image:: ./images/double_gaussian_2d.png
    :align: center

Trace plot
----------

.. image:: ./images/double_gaussian_trace.png
    :align: center

Run plot
--------

.. image:: ./images/double_gaussian_run.png
    :align: center

Rosenbrock
==========

The ``Rosenbrock`` distribution is one of the most infamous sampling and optimisation
targets. The reason is clear and it is its highly warped geometry. Here we sample from
this very challenging target in 10D extremely efficiently.

2D marginal posterior
---------------------

.. image:: ./images/rosenbrock_2d.png
    :align: center

Trace plot
----------

.. image:: ./images/rosenbrock_trace.png
    :align: center

Run plot
--------

.. image:: ./images/rosenbrock_run.png
    :align: center

Funnel
======

``Neal's funnel`` as it is most commonly known is a very challenging target distribution
that makes sampling from it using common MCMC methods such as Hamiltonian Monte Carlo a
very difficult task. ``pocoMC`` manages to sample very efficiently by internally decorrelating
its geometry, thus simplifying the problem.

2D marginal posterior
---------------------

.. image:: ./images/funnel_2d.png
    :align: center

Trace plot
----------

.. image:: ./images/funnel_trace.png
    :align: center

Run plot
--------

.. image:: ./images/funnel_run.png
    :align: center