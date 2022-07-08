import numpy as np
import math
import torch
from tqdm import tqdm
import warnings

SQRTEPS = math.sqrt(float(np.finfo(np.float64).eps))


def get_ESS(logw: np.ndarray):
    # TODO rename to get_ess or compute_ess
    logw_max = np.max(logw)
    logw_normed = logw - logw_max

    weights = np.exp(logw_normed) / np.sum(np.exp(logw_normed))
    return 1.0 / np.sum(weights * weights) / len(weights)


def resample_equal(samples: np.ndarray,
                   weights: np.ndarray,
                   rstate: np.random.RandomState = None):
    """
        Resample a new set of points from the weighted set of inputs
        such that they all have equal weight.
        Each input sample appears in the output array either
        `floor(weights[i] * nsamples)` or `ceil(weights[i] * nsamples)` times,
        with `floor` or `ceil` randomly selected (weighted by proximity).

    Parameters
    ----------
    samples : `~numpy.ndarray` with shape (nsamples,)
        Set of unequally weighted samples.
    weights : `~numpy.ndarray` with shape (nsamples,)
        Corresponding weight of each sample.
    rstate : `~numpy.random.RandomState`, optional
        `~numpy.random.RandomState` instance.

    Returns
    -------
    equal_weight_samples : `~numpy.ndarray` with shape (nsamples,)
        New set of samples with equal weights.
    
    Examples
    --------
    >>> x = np.array([[1., 1.], [2., 2.], [3., 3.], [4., 4.]])
    >>> w = np.array([0.6, 0.2, 0.15, 0.05])
    >>> resample_equal(x, w)
    array([[ 1.,  1.],
           [ 1.,  1.],
           [ 1.,  1.],
           [ 3.,  3.]])
    
    Notes
    -----
    Implements the systematic resampling method described in `Hol, Schon, and
    Gustafsson (2006) <doi:10.1109/NSSPW.2006.4378824>`_.
    """

    if rstate is None:
        rstate = np.random

    if abs(np.sum(weights) - 1.) > SQRTEPS:  # same tol as in np.random.choice.
        # Guarantee that the weights will sum to 1.
        # warnings.warn("Weights do not sum to 1 and have been renormalized.")
        weights = np.array(weights) / np.sum(weights)

    # Make N subdivisions and choose positions with a consistent random offset.
    n_samples = len(weights)
    positions = (rstate.random() + np.arange(n_samples)) / n_samples

    # Resample the data.
    idx = np.zeros(n_samples, dtype=int)
    cumulative_sum = np.cumsum(weights)
    i, j = 0, 0
    while (i < n_samples) and (j < len(cumulative_sum)):
        if positions[i] < cumulative_sum[j]:
            idx[i] = j
            i += 1
        else:
            j += 1

    return samples[idx]


class ProgressBar:
    def __init__(self, show: bool = True):
        """
        TODO write docstring.

        Parameters
        ----------
        show
        """
        self.progress_bar = tqdm(desc='Iter', disable=not show)
        self.info = dict()

    def update_stats(self, info):
        """
        TODO write docstring.

        Parameters
        ----------
        info

        Returns
        -------

        """
        self.info = {**self.info, **info}
        self.progress_bar.set_postfix(ordered_dict=self.info)

    def update_iter(self):
        """
        TODO write docstring.

        Returns
        -------

        """
        self.progress_bar.update(1)

    def close(self):
        """
        TODO write docstring.

        Returns
        -------

        """
        self.progress_bar.close()


class _FunctionWrapper(object):
    r"""
        This is a hack to make the likelihood function pickleable when ``args``
        or ``kwargs`` are also included.

    Parameters
    ----------
    f : (callable)
        Log Probability function.
    args : list
        Extra arguments to be passed into the logprob.
    kwargs : dict
        Extra arguments to be passed into the logprob.

    Returns
    -------
        Log Probability function.
    """

    def __init__(self, f, args, kwargs):
        self.f = f
        self.args = [] if args is None else args
        self.kwargs = {} if kwargs is None else kwargs

    def __call__(self, x):
        return self.f(x, *self.args, **self.kwargs)


def torch_to_numpy(x):
    return x.detach().numpy()


def numpy_to_torch(x):
    return torch.tensor(x, dtype=torch.float32)


def torch_double_to_float(x, warn=True):
    if x.dtype == torch.float64 and warn:
        warnings.warn(f"Float64 data is currently unsupported, casting to Float32. Output will also have type Float32.")
        return x.float()
    elif x.dtype == torch.float32:
        return x
    else:
        raise ValueError(f"Unsupported datatype for input data: {x.dtype}")
