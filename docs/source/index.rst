.. pocoMC documentation master file, created by
   sphinx-quickstart on Fri Apr 29 13:25:54 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. title:: pocoMC documentation

.. figure:: ./../../logo.png
    :scale: 30 %
    :align: center

**pocoMC** is a Python implementation of Preconditioned Monte Carlo (PMC). Using
`pocoMC` one can perform Bayesian inference, including model comparison, for
challenging scientific problems. `pocoMC`'s sophisticated normalizing flow
preconditioning procedures enables sampling from highly correlated posterior
distributions. `pocoMC` is designed to excel in demanding parameter estimation
problems that include multimodal and highly non--Gaussian target distributions.

.. admonition:: Where to start?
    :class: tip

    🖥 A good place to get started is with the {ref}`install` and then the
    {ref}`tutorials`.

    📖 For all the details, check out the {ref}`guide`, including the [full API
    documentation](api-ref).

    💡 If you're running into problems getting `pocoMC` to do what you want, first
    check out the {ref}`troubleshooting` page, for some general tips and tricks.

    🐛 If {ref}`troubleshooting` doesn't solve your problems, or if you find bugs,
    check out the {ref}`contributing` and then head on over to the [GitHub issues
    page](https://github.com/minaskar/pocomc/issues).

    👈 Check out the sidebar to find the full table of contents.



.. toctree::
   :hidden:
   :maxdepth: 4
   :caption: Contents:

   install
   pages/quickstart.ipynb
   tutorials
   api
   GitHub Repository <https://github.com/minaskar/pocomc>



Attribution
===========

Please cite the following if you find this code useful in your
research. The BibTeX entry for the paper is::

    @article{x,
        title={x},
        author={x},
        year={x},
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
