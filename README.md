![logo](logo.png)

**pocoMC is a Python implementation of the Preconditioned Monte Carlo method for accelerated Bayesian inference**

## Example

For instance, if you wanted to draw samples from a 10-dimensional Rosenbrock distribution with a uniform prior, you would do something like:

```python
import pocomc as pc
import numpy as np

ndim = 10  # Number of dimensions

lower = np.full(ndim, -10.) # lower bound of the prior
upper = np.full(ndim, 10.) # upper bound of the prior
bounds = np.c_[lower, upper]
const = np.sum(np.log(upper - lower))  # log of the Uniform density

def log_prior(x):
    if np.any((x < lower) | (x > upper)):  # If any dimension is out of bounds, the log prior is -infinity
        return -np.inf 
    else:
        return -const

def log_likelihood(x):
    return -np.sum(10.0*(x[:,::2]**2.0 - x[:,1::2])**2.0 \
            + (x[:,::2] - 1.0)**2.0, axis=1)


nwalkers = 1000
prior_samples = np.random.uniform(size=(nwalkers, ndim), low=-10.0, high=10.0)

sampler = pc.Sampler(nwalkers,
                     ndim,
                     log_likelihood,
                     log_prior,
                     vectorize_likelihood=True,
                     bounds=bounds
                    )
sampler.run(prior_samples)

results = sampler.results # Dictionary with results
```

## Documentation

Read the docs at [pocomc.readthedocs.io](https://pocomc.readthedocs.io)


## Installation

To install ``pocomc`` using ``pip`` run:

```bash
pip install pocomc
```

## Attribution

Please cite the following papers if you found this code useful in your research:

```bash
@article{karamanis2022pocomc,
  title={Accelerating astronomical and cosmological inference with Preconditioned Monte Carlo},
  author={Karamanis, Minas and Beutler, Florian and Peacock, John A and Nabergoj, David, and Seljak, Uro\v{s}},
  journal={in prep},
  year={2022}
}
```

## Licence

Copyright 2022-Now Minas Karamanis and contributors.

zeus is free software made available under the GPL-3.0 License. For details see the `LICENSE` file.
