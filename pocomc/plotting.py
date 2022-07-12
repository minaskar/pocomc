from typing import List, Union

import numpy as np


def trace(results: dict,
          labels: List = None,
          dims: Union[List, np.ndarray] = None,
          width: float = 10.0,
          height: float = 3.0,
          kde_bins: int = 200):
    r"""Plot particle traces and 1-D marginal posteriors.

    Parameters
    ----------
    results : dict
        Result dictionary produced using ``pocoMC``.
    labels : list or None
        List of parameter names to include in the figure. If ``None``, the labels are set automatically.  Default: None.
    dims : list or `np.ndarray` or None
        The subset of dimensions that should be plotted. If None, all dimensions will be shown. Default: None.
    width : float
        Width of figure. Default: ``10.0``.
    height : float
        Height of each subplot. Default: ``3.0``.
    kde_bins : int
        Number of bins to use for KDE. Default: ``200``.
    
    Returns
    -------
    fig : Figure
        Trace figure.
    """

    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde

    # Get samples and log-weights
    samples = results.get("x")
    logw = results.get("logw")
    beta = results.get("beta")

    # Compute weights
    weights = np.exp(logw - np.max(logw, axis=1)[:, np.newaxis])
    weights /= np.sum(weights, axis=1)[:, np.newaxis]

    # Number of beta values, particles and parameters/dimensions
    n_beta, n_particles, n_dim = np.shape(samples)

    # Set labels if None is provided
    if labels is None:
        if dims is None:
            labels = [r"$x_{%s}$" % i for i in range(n_dim)]
        else:
            labels = [r"$x_{%s}$" % i for i in dims]

    if dims is None:
        parameters = np.arange(n_dim)
    else:
        parameters = dims
        n_dim = len(dims)

    # Initialise figure
    fig = plt.figure(figsize=(width, n_dim * height))

    # Loop over parameters/dimensions
    for i, p in enumerate(parameters):
        # Left column -- beta plots
        plt.subplot(n_dim, 2, 2 * i + 1)
        for j in range(n_beta):
            plt.scatter(
                np.full(n_particles, beta[j]),
                samples[j, :, p],
                s=5,
                c='C0',
                alpha=0.1 * weights[j] / np.max(weights[j])
            )
        plt.xscale('log')
        plt.xlabel(r'$\beta$', fontsize=14)
        plt.ylabel(labels[i], fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)

        # Compute 1D KDE for parameter for beta = 1.0
        kde = gaussian_kde(samples[-1, :, p], weights=weights[-1])
        x = np.linspace(np.min(samples[-1, :, p]), np.max(samples[-1, :, p]), kde_bins)

        # Right column -- trace plots
        plt.subplot(n_dim, 2, 2 * i + 2)
        plt.fill_between(x, kde(x), alpha=0.8)
        plt.xlabel(labels[i], fontsize=14)
        plt.ylabel('')
        plt.xticks(fontsize=12)
        plt.yticks([])

    plt.tight_layout()
    return fig


def corner(results: dict,
           labels: List = None,
           dims: Union[List, np.ndarray] = None,
           color: str = None,
           **corner_kwargs):
    r"""Corner plot of the 1-D and 2-D marginal posteriors.

    Parameters
    ----------
    results : dict
        Result dictionary produced using ``pocoMC``.
    labels : list or None
        List of parameter names to include in the figure. If ``labels=None``
        (default) the labels are set automatically.
    dims : list or `np.ndarray` or None
        The subset of dimensions that should be plotted. If not provided, all dimensions will be shown.
    color : str or None
        Color to use for contours.
    **corner_kwargs : -
        Any additional arguments passed to ``corner.corner``.
        See here ``https://corner.readthedocs.io/en/latest/api.html``
        for a complete list.
    
    Returns
    -------
    fig : Figure
        Corner figure.
    """
    import corner

    if color is None:
        color = 'C0'

    # Get posterior samples
    posterior_samples = results.get("samples")

    # Number of  particles and parameters/dimensions
    n_particles, n_dim = np.shape(posterior_samples)

    # Set labels if None is provided
    if labels is None:
        if dims is None:
            labels = [r"$x_{%s}$" % i for i in range(n_dim)]
        else:
            labels = [r"$x_{%s}$" % i for i in dims]

    if dims is not None:
        posterior_samples = posterior_samples[:, dims]

    return corner.corner(data=posterior_samples,
                         labels=labels,
                         color=color,
                         **corner_kwargs)


def run(results: dict,
        full_run: bool = True,
        width: float = 10.0,
        height: float = 10.0):
    r"""Run plot which shows various statistics computed during the run.

    Parameters
    ----------
    results : dict
        Result dictionary produced using ``pocoMC``.
    full_run : bool
        If True, include run diagnostics beyond the basic run. Default: ``True``.
    width : float
        Width of figure. Default: ``10.0``.
    height : float
        Height of each subplot. Default: ``10.0``.
    
    Returns
    -------
    fig : Figure
        Run figure.
    """
    beta = results.get("beta")
    steps = results.get("steps")
    scale = results.get("scale")
    logz = results.get("logz")

    if full_run:
        if len(steps) > len(beta):
            beta = np.hstack((beta, beta[-1] * np.ones(len(steps) - len(beta))))

        if len(steps) > len(logz):
            logz = np.hstack((logz, logz[-1] * np.ones(len(steps) - len(logz))))
    else:
        steps = steps[:len(beta)]
        scale = scale[:len(beta)]

    import matplotlib.pyplot as plt

    # Initialise figure
    fig = plt.figure(figsize=(width, height))

    plt.subplot(4, 1, 1)
    plt.plot(beta, 'o-', lw=2.5)
    plt.ylabel(r"$\beta$", fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    plt.subplot(4, 1, 2)
    plt.plot(steps, 'o-', lw=2.5)
    plt.ylabel("steps", fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    plt.subplot(4, 1, 3)
    plt.plot(scale, 'o-', lw=2.5)
    plt.ylabel("scale", fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    plt.subplot(4, 1, 4)
    plt.plot(logz, 'o-', lw=2.5)
    plt.ylabel(r"$\log \mathcal{Z}$", fontsize=16)
    plt.xlabel("Iterations", fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    plt.tight_layout()

    return fig
