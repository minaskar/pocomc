__bibtex__ = """
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
"""
__url__ = "https://pocomc.readthedocs.io"
__author__ = "Minas Karamanis"
__email__ = "minaskar@gmail.com"
__license__ = "GPL-3.0"
__description__ = "A Python implementation of Preconditioned Monte Carlo for accelerated Bayesian Computation"


from .flow import *
from .sampler import *
from .prior import *
from .parallel import *
from ._version import version

__version__ = version
