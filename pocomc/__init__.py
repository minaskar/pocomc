__bibtex__ = """
@article{karamanis2022pocomc,
  title={Accelerating astronomical and cosmological inference with Preconditioned Monte Carlo},
  author={Karamanis, Minas and Beutler, Florian and Peacock, John A and Nabergoj, David, and Seljak, Uro\v{s}},
  journal={in prep},
  year={2022}
}
"""
__url__ = "https://pocomc.readthedocs.io"
__author__ = "Minas Karamanis"
__email__ = "minaskar@gmail.com"
__license__ = "GPL-3.0"
__description__ = "A Python implementation of Preconditioned Monte Carlo for accelerated Bayesian Computation"


from .flow import *
from .sampler import *
from .plotting import *
from ._version import version

__version__ = version
