import numpy as np 
import math
import torch
from tqdm import tqdm

SQRTEPS = math.sqrt(float(np.finfo(np.float64).eps))


def get_ESS_CV(logw):

    logw_max = np.max(logw)
    logw_normed = logw - logw_max

    weights = np.exp(logw_normed) / np.sum(np.exp(logw_normed))

    return 1.0 / np.sum(weights*weights), np.std(weights)/np.mean(weights)


def resample_equal(samples, weights, rstate=None):
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
    >>> utils.resample_equal(x, w)
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
        #warnings.warn("Weights do not sum to 1 and have been renormalized.")
        weights = np.array(weights) / np.sum(weights)

    # Make N subdivisions and choose positions with a consistent random offset.
    nsamples = len(weights)
    positions = (rstate.random() + np.arange(nsamples)) / nsamples

    # Resample the data.
    idx = np.zeros(nsamples, dtype=np.int)
    cumulative_sum = np.cumsum(weights)
    i, j = 0, 0
    while i < nsamples:
        if positions[i] < cumulative_sum[j]:
            idx[i] = j
            i += 1
        else:
            j += 1

    return samples[idx]


class progress_bar:

    def __init__(self, show=True):
        self.show = show
        if self.show:
            self.progress_bar = tqdm(desc='Iter')

    def update(self, beta, ncall, ESS, CV, logz, extra=None):
        if self.show:
            self.progress_bar.update(1)
            if extra is None:
                self.progress_bar.set_postfix(ordered_dict={'beta':np.round(beta,8),
                                                            'ncall':int(ncall),
                                                            'ESS':int(ESS),
                                                            'CV':np.round(CV,3),
                                                            'logz':np.round(logz,4)})
            else:
                self.progress_bar.set_postfix(ordered_dict={'beta':np.round(beta,8),
                                                            'ncall':int(ncall),
                                                            'ESS':int(ESS),
                                                            'CV':np.round(CV,3),
                                                            'logz':np.round(logz,4),
                                                            'extra':extra})

    def close(self):
        if self.show:
            self.progress_bar.close()


class _FunctionWrapper(object):
    """
    This is a hack to make the likelihood function pickleable when ``args``
    or ``kwargs`` are also included.

    Args:
        f (callable) : Log Probability function.
        args (list): Extra arguments to be passed into the logprob.
        kwargs (dict): Extra arguments to be passed into the logprob.

    Returns:
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