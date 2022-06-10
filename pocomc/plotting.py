import numpy as np

def trace(results,
          labels=None,
          width=10.0,
          height=3.0,
          kde_bins=200,
         ):
    r"""Trace plot.

    Parameters
    ----------
    results : dict
        Result dictionary produced using ``pocoMC``.
    labels : list or None
        List of parameter names to include in the figure. If ``labels=None``
        (default) the labels are set automatically.
    width : float
        width of figure (default is ``width=10.0``).
    height : float
        Height of each subplot (default is ``height=3.0``).
    kde_bins : int
        Number of bins to use for KDE (default is ``kde_bins=200``).
    
    """

    # import plt and scipy
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde

    # Get samples and log-weights
    samples = results.get("samples")
    logw = results.get("logw")
    beta = results.get("beta")

    # Compute weights
    weights = np.exp(logw-np.max(logw,axis=1)[:,np.newaxis])
    weights /= np.sum(weights, axis=1)[:,np.newaxis]

    # Number of beta values, particles and parameters/dimensions
    n_beta, n_particles, n_dim = np.shape(samples)

    # Set labels if None is provided
    if labels is None:
        labels = [r"$x_{%s}$"%i for i in range(n_dim)]

    # Initialise figure
    fig = plt.figure(figsize=(width, n_dim * height))

    # Loop over parameters/dimensions
    for i in range(n_dim):
        # Left column -- beta plots
        plt.subplot(n_dim, 2, 2 * i + 1)
        for j in range(n_beta):
            plt.scatter(np.full(n_particles, beta[j]),
                        samples[j,:,i],
                        s=5,
                        c='C0',
                        alpha=0.1*weights[j]/np.max(weights[j]))
        plt.xscale('log')
        plt.xlabel(r'$\beta$', fontsize=14)
        plt.ylabel(labels[i], fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)

        # Compute 1D KDE for parameter for beta = 1.0
        kde = gaussian_kde(samples[-1,:,i], weights=weights[-1])
        x = np.linspace(np.min(samples[-1,:,i]), np.max(samples[-1,:,i]), kde_bins)

        # Right column -- trace plots
        plt.subplot(n_dim, 2, 2 * i + 2)
        plt.fill_between(x, kde(x), alpha=0.8)
        plt.xlabel(labels[i], fontsize=14)
        plt.ylabel('')
        plt.xticks(fontsize=12)
        plt.yticks([])
    
    plt.tight_layout()
    return fig


def corner(results,
           labels=None,
           color='C0',
           bins=20,
           range=None,
           smooth=None,
           smooth1d=None,
           titles=None,
           show_titles=True,
           title_quantiles=None,
           ):
    r"""Corner plot.

    Parameters
    ----------
    results : dict
        Result dictionary produced using ``pocoMC``.
    """
    # import corner
    import corner

    # Get posterior samples
    posterior_samples = results.get("posterior_samples")

    # Number of  particles and parameters/dimensions
    n_particles, n_dim = np.shape(posterior_samples)

    # Set labels if None is provided
    if labels is None:
        labels = [r"$x_{%s}$"%i for i in range(n_dim)]
    
    return corner.corner(data=posterior_samples,
                         labels=labels,
                         color=color,
                         bins=bins,
                         range=range,
                         smooth=smooth,
                         smooth1d=smooth1d,
                         titles=titles,
                         show_titles=show_titles,
                         title_quantiles=title_quantiles,


                        )


def run(results):
    r"""Run plot.

    Parameters
    ----------
    results : dict
        Result dictionary produced using ``pocoMC``.
    """
    pass