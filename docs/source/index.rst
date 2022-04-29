.. pocoMC documentation master file, created by
   sphinx-quickstart on Fri Apr 29 13:25:54 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. title:: pocoMC documentation

.. figure:: ./../../logo.png
    :scale: 30 %
    :align: center

``pocoMC`` is a Python implementation of `Preconditioned Monte Carlo (PMC)`.
Using ``pocoMC`` one can perform Bayesian inference, including model comparison, 
for challenging scientific problems. ``pocoMC``'s sophisticated `normalizing flow
preconditioning` procedures enable efficient sampling from highly correlated posterior
distributions. ``pocoMC`` is designed to excel in demanding parameter estimation
problems that include multimodal and highly non--Gaussian target distributions.

.. admonition:: Where to start?
    :class: tip

    🖥 A good place to get started is with the :doc:`install` and then the
    :doc:`pages/quickstart` guide.

    📖 For more details, check out the :doc:`tutorials`, including the full :doc:`api`
    documentation.

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
   tutorials
   faq
   api
   GitHub Repository <https://github.com/minaskar/pocomc>



Attribution & Citation
======================

Please cite the following if you find this code useful in your
research. The BibTeX entry for the paper is::

    @article{x,
        title={x},
        author={x},
        year={2022},
        note={in prep}
    }


Authors & License
=================

Copyright 2022 Minas Karamanis and contributors.

``pocoMC`` is free software made available under the ``GPL-3.0 License``.


Changelog
=========

**0.0.1 (30/04/22)**

- First version
