![logo](logo.png)

**pocoMC is a Python implementation of the Preconditioned Monte Carlo method for accelerated Bayesian inference**

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://github.com/minaskar/pocomc/blob/master/LICENSE)
[![Documentation Status](https://readthedocs.org/projects/pocomc/badge/?version=latest)](https://pocomc.readthedocs.io/en/latest/?badge=latest)
[![build](https://github.com/minaskar/pocomc/actions/workflows/setup_and_run_tests.yml/badge.svg)](https://github.com/minaskar/pocomc/actions/workflows/setup_and_run_tests.yml)


# Getting started

## Brief introduction

``pocoMC`` utilises a *Normalising Flow* in order to precondition the target distribution by removing any correlations between its parameters. The code then generates posterior samples, that can be used for parameter estimation, using a powerful adaptive *Sequential Monte Carlo* algorithm manifesting a sampling effiency that can be orders of magnitude higher than without precondition. Furthermore, ``pocoMC`` also provides an unbiased estimate of the *model evidence* that can be used for the task of *Bayesian model comparison*.

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

For instance, if you wanted to draw samples from a 10-dimensional Rosenbrock distribution with a uniform prior, you would do something like:

```python
import pocomc as pc
import numpy as np

n_dim = 10  # Number of dimensions

def log_prior(x):
    if np.any((x < -10.0) | (x > 10.0)):  # If any dimension is out of bounds, the log prior is -infinity
        return -np.inf 
    else:
        return 0.0

def log_likelihood(x):
    return -np.sum(10.0*(x[:,::2]**2.0 - x[:,1::2])**2.0 \
            + (x[:,::2] - 1.0)**2.0, axis=1)


n_particles = 1000
prior_samples = np.random.uniform(size=(n_particles, n_dim), low=-10.0, high=10.0)

sampler = pc.Sampler(
    n_particles,
    n_dim,
    log_likelihood,
    log_prior,
    vectorize_likelihood=True,
    bounds=(-10.0, 10.0)
)
sampler.run(prior_samples)

results = sampler.results # Dictionary with results
```


# Attribution & Citation

Please cite the following papers if you found this code useful in your research:

```bash
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
```

# Licence

Copyright 2022-Now Minas Karamanis and contributors.

``pocoMC`` is free software made available under the GPL-3.0 License. For details see the `LICENSE` file.
