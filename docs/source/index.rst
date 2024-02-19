.. pocoMC documentation master file, created by
   sphinx-quickstart on Fri Apr 29 13:25:54 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

|

.. title:: pocoMC documentation

.. figure:: ./../../logo.svg
    :scale: 30 %
    :align: center

|

pocoMC
======

``pocoMC`` is a Python implementation of `Preconditioned Monte Carlo (PMC)`.
Using ``pocoMC`` one can perform Bayesian inference, including model comparison, 
for challenging scientific problems. ``pocoMC``'s sophisticated `normalizing flow
preconditioning` procedures enables efficient sampling from highly correlated posterior
distributions. ``pocoMC`` is designed to excel in demanding parameter estimation
problems that include multimodal and highly non--Gaussian target distributions.

.. admonition:: Where to start?
    :class: tip

    🖥 A good place to get started is with the :doc:`install` and then the
    :doc:`pages/quickstart` guide. If you are not familiar with Bayesian inference
    have a look at the :doc:`background`.

    📖 For more details, check out the :doc:`advanced` and :doc:`tutorials`,
    including the full :doc:`api` documentation.

    💡 If you're running into problems getting ``pocoMC`` to do what you want, first
    check out the :doc:`faq` page, for some general tips and tricks.

    🐛 If :doc:`faq` doesn't solve your problems, or if you find bugs,
    then head on over to the `GitHub issues page <https://github.com/minaskar/pocomc/issues>`_.

    👈 Check out the sidebar to find the full table of contents.



.. toctree::
   :hidden:
   :maxdepth: 4
   :caption: Contents:

   install
   pages/quickstart.ipynb
   background
   advanced
   tutorials
   faq
   api
   GitHub Repository <https://github.com/minaskar/pocomc>



Attribution & Citation
======================

Please cite the following if you find this code useful in your
research. The BibTeX entries for the papers are::

    @article{karamanis2022accelerating,
        title={Accelerating astronomical and cosmological inference with preconditioned Monte Carlo},
        author={Karamanis, Minas and Beutler, Florian and Peacock, John A and Nabergoj, David and Seljak, Uro{\v{s}}},
        journal={Monthly Notices of the Royal Astronomical Society},
        volume={516},
        number={2},
        pages={1644--1653},
        year={2022},
        publisher={Oxford University Press}
    }

    @article{karamanis2022pocomc,
        title={pocoMC: A Python package for accelerated Bayesian inference in astronomy and cosmology},
        author={Karamanis, Minas and Nabergoj, David and Beutler, Florian and Peacock, John A and Seljak, Uros},
        journal={arXiv preprint arXiv:2207.05660},
        year={2022}
    }


Authors & License
=================

Copyright 2022-2024 Minas Karamanis and contributors.

``pocoMC`` is free software made available under the ``GPL-3.0 License``.


Changelog
=========

**1.0.2 (18/02/24)**

- Minor improvements and bug fixes.

**1.0.0 (28/01/24)**

- First stable release.
- Major refactoring of the code.
- Added support for multiple normalizing flows through ``zuko``.
- Added preconditioned Crank-Nicolson sampler.
- Added support for multilayer SMC.

**0.2.2 (22/08/22)**

- Fixed bridge sampling estimator.
- Improved likelihood call counter.

**0.1.2 (27/07/22)**

- Bridge sampling estimator for the model evidence.
- Added probit transform for bounded parameters.

**0.1.11 (12/07/22)**

- Include saving and loading the state of the sampler. Useful for resuming runs from files.

**0.1.0 (12/07/22)**

- First version
