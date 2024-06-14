![logo](logo.png)

**pocoMC is a Python implementation of the Preconditioned Monte Carlo method for accelerated Bayesian inference**

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://github.com/minaskar/pocomc/blob/master/LICENSE)
[![Documentation Status](https://readthedocs.org/projects/pocomc/badge/?version=latest)](https://pocomc.readthedocs.io/en/latest/?badge=latest)


# Getting started

## Brief introduction

``pocoMC`` is a Python package for fast Bayesian posterior and model evidence estimation. It leverages 
the Preconditioned Monte Carlo (PMC) algorithm, offering significant speed improvements over 
traditional methods like MCMC and Nested Sampling. Ideal for large-scale scientific problems 
with expensive likelihood evaluations, non-linear correlations, and multimodality, ``pocoMC`` 
provides efficient and scalable posterior sampling and model evidence estimation. Widely used 
in cosmology and astronomy, ``pocoMC`` is user-friendly, flexible, and actively maintained.

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
from scipy.stats import uniform

n_dim = 10  # Number of dimensions

prior = pc.Prior(n_dim*[uniform(-10.0, 20.0)]) # U(-10,10)

def log_likelihood(x):
    return -np.sum(10.0*(x[:,::2]**2.0 - x[:,1::2])**2.0 \
            + (x[:,::2] - 1.0)**2.0, axis=1)

sampler = pc.Sampler(
    prior=prior,
    likelihood=log_likelihood,
    vectorize=True,
)
sampler.run()

samples, weights, logl, logp = sampler.posterior() # Weighted posterior samples

logz, logz_err = sampler.evidence() # Bayesian model evidence estimate and uncertainty
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
