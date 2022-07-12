![logo](logo.png)

**pocoMC is a Python implementation of the Preconditioned Monte Carlo method for accelerated Bayesian inference**

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://github.com/minaskar/pocomc/blob/master/LICENSE)
[![Documentation Status](https://readthedocs.org/projects/pocomc/badge/?version=latest)](https://pocomc.readthedocs.io/en/latest/?badge=latest)
[![build](https://github.com/minaskar/pocomc/actions/workflows/setup_and_run_tests.yml/badge.svg)](https://github.com/minaskar/pocomc/actions/workflows/setup_and_run_tests.yml)

# Getting started

## Brief introduction

``pocoMC`` utilises a *Normalising Flow* in order to precondition the target distribution by removing any correlations
between its parameters. The code then generates posterior samples, that can be used for parameter estimation, using a
powerful adaptive *Sequential Monte Carlo* algorithm manifesting a sampling effiency that can be orders of magnitude
higher than without precondition. Furthermore, ``pocoMC`` also provides an unbiased estimate of the *model evidence*
that can be used for the task of *Bayesian model comparison*.

## Documentation

Read the docs at [pocomc.readthedocs.io](https://pocomc.readthedocs.io) for more information, examples and tutorials.

## Installation

To install ``pocomc`` using ``pip`` run:

```bash
pip install pocomc
```

or, to install from source:

```bash
git clone https://github.com/minaskar/pocomc.git
cd pocomc
python setup.py install
```

## Basic example

For instance, if you wanted to draw samples from a 10-dimensional Rosenbrock distribution with a uniform prior, you
would do something like:

```python
import pocomc as pc
import numpy as np
from pocomc.priors import Uniform

n_dim = 10  # Number of dimensions


def log_likelihood(x):
    return -np.sum(10.0 * (x[:, ::2] ** 2.0 - x[:, 1::2]) ** 2.0 + (x[:, ::2] - 1.0) ** 2.0, axis=1)


n_walkers = 1000
sampler = pc.Sampler(
    n_walkers,
    n_dim,
    log_likelihood,
    Uniform(-10, 10, n_dim),
    vectorize_likelihood=True
)
sampler.run()

results = sampler.results  # Dictionary with results
```

# Attribution & Citation

Please cite the following papers if you found this code useful in your research:

```bash
@article{karamanis2022pmc,
    title={Accelerating astronomical and cosmological inference with Preconditioned Monte Carlo},
    author={Karamanis, Minas and Beutler, Florian and Peacock, John A and Nabergoj, David, and Seljak, Uro\v{s}},
    journal={in prep},
    year={2022}
}

@article{karamanis2022pocomc,
    title={pocoMC: A Python package for accelerated Bayesian inference in astronomy and cosmology},
    author={Karamanis, Minas and Nabergoj, David, and Beutler, Florian and Peacock, John A and Seljak, Uro\v{s}},
    journal={in prep},
    year={2022}
}
```

# Licence

Copyright 2022-Now Minas Karamanis and contributors.

``pocoMC`` is free software made available under the GPL-3.0 License. For details see the `LICENSE` file.
