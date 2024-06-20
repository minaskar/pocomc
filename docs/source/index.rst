|

.. title:: pocoMC documentation

.. figure:: ./images/logo.gif
    :scale: 100 %
    :align: center

|

    ``pocoMC`` is a Python package for fast Bayesian posterior and model evidence estimation. It leverages 
    the Preconditioned Monte Carlo (PMC) algorithm, offering significant speed improvements over 
    traditional methods like MCMC and Nested Sampling. Ideal for large-scale scientific problems 
    with expensive likelihood evaluations, non-linear correlations, and multimodality, ``pocoMC`` 
    provides efficient and scalable posterior sampling and model evidence estimation. Widely used 
    in cosmology and astronomy, ``pocoMC`` is user-friendly, flexible, and actively maintained.

.. admonition:: Where to start?
    :class: tip

    🖥 A good place to get started is with the :doc:`install` and then the
    :doc:`quickstart` guide. If you are not familiar with Bayesian inference
    have a look at the :doc:`background`.

    📖 For more details, check out the :doc:`likelihood` through :doc:`blobs` information, 
    as well as the :doc:`fitting` and :doc:`model_comparison` tutorials.

    💡 If you're running into problems getting ``pocoMC`` to do what you want, first
    check out the :doc:`faq` page, for some general tips and tricks.

    🐛 If :doc:`faq` doesn't solve your problems, or if you find bugs,
    then head on over to the `GitHub issues page <https://github.com/minaskar/pocomc/issues>`_.

    👈 Check out the sidebar to find the full table of contents.



.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: User Guide:

   install
   quickstart.ipynb
   likelihood.ipynb
   priors
   sampling.ipynb
   results
   parallelization.ipynb
   flow.ipynb
   checkpoint.ipynb
   blobs.ipynb


.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Tutorials:

   fitting
   model_comparison

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Discussion:

   background
   faq

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: API Documentation:

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

**1.2.2 (20/06/24)**

- Fixed bug in ``posterior`` method related to blobs.

**1.2.1 (14/06/24)**

- Added support for log-likelihoods that return ``-np.inf`` inside the prior volume.

**1.2.0 (11/06/24)**

- Added ``MPIPool`` for parallelization.
- Fixed bugs in checkpointing when using MPI in NFS4 and BeeGFS filesystems.
- Automatically save final checkpoint file when finishing the run if ``save_every`` is not ``None``.
- Added option to continue sampling after completing the run.

**1.1.0 (31/05/24)**

- Fix robustness issues with the Crank-Nicolson sampler.
- Added predefined normalizing flows.
- Added support for derived parameters through the ``blobs`` framework.
- Added ``dynamic`` mode for determining the ESS based on the unique sample size (USS).
- Added internal ``multiprocess`` pool for parallelization.
- Improved documentation and tutorials.

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
