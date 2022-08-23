import numpy as np
import torch

from .tools import numpy_to_torch, torch_to_numpy


class Pearson:
    """
    Pearson correlation coefficient.
    This is a measure of the linear correlation between
    two sets of particles. This is used to adaptively
    determine the number of MCMC steps required per iteration
    by comparing the initial particle distribution with
    the current particle distribution and terminating
    MCMC once the correlation coefficient drops below
    a threshold value.

    Parameters
    ----------
    a : ``np.ndarray`` of shape ``(nparticles, ndim)``
        Initial positions of particles
    """

    def __init__(self, a):
        self.l = a.shape[0]  # TODO avoid ambiguous variable name 'l', rename to something else.
        self.am = a - np.sum(a, axis=0) / self.l
        self.aa = np.sum(self.am ** 2, axis=0) ** 0.5

    def get(self, b):
        """
        Method that computes the correlation coefficients between current positions and initial.

        Parameters
        ----------
        b : ``np.ndarray`` of shape ``(n_particles, n_dim)``
            Current positions of particles
        Returns
        -------
        correlation coefficients
        """
        bm = b - np.sum(b, axis=0) / self.l
        bb = np.sum(bm ** 2, axis=0) ** 0.5
        ab = np.sum(self.am * bm, axis=0)
        return np.abs(ab / (self.aa * bb))


@torch.no_grad()
def preconditioned_metropolis(state_dict: dict,
                              function_dict: dict,
                              option_dict: dict):
    """
    Preconditioned Metropolis
    Function that samples the current target distribution
    using a simple Random-walk Metropolis algorithm on the
    latent (i.e. uncorrelated or preconditioned) parameter
    space. The lack of correlations renders Preconditioned
    Metropolis more efficient that standard Metropolis.
    
    Parameters
    ----------
    state_dict : dict
        Dictionary of current state
    function_dict : dict
        Dictionary of functions.
    option_dict : dict
        Dictionary of options.
    
    Returns
    -------
    Results dictionary
    """
    # Likelihood call counter
    n_calls = 0

    # Clone state variables
    u = torch.clone(numpy_to_torch(state_dict.get('u')))
    x = torch.clone(numpy_to_torch(state_dict.get('x')))
    J = torch.clone(numpy_to_torch(state_dict.get('J')))
    L = torch.clone(numpy_to_torch(state_dict.get('L')))
    P = torch.clone(numpy_to_torch(state_dict.get('P')))
    beta = state_dict.get('beta')
    Z = numpy_to_torch(log_prob(torch_to_numpy(L), torch_to_numpy(P), beta))

    # Get functions
    log_like = function_dict.get('loglike')
    log_prior = function_dict.get('logprior')
    scaler = function_dict.get('scaler')
    flow = function_dict.get('flow')

    # Get MCMC options
    n_min = option_dict.get('nmin')
    n_max = option_dict.get('nmax')
    sigma = option_dict.get('sigma')
    corr_threshold = option_dict.get('corr_threshold')
    progress_bar = option_dict.get('progress_bar')

    # Get number of particles and parameters/dimensions
    n_walkers, n_dim = x.shape

    # Transform u to theta
    theta, log_det_J = flow.forward(u)
    J_flow = -log_det_J.sum(-1)

    # Initialise Pearson correlation object
    corr = Pearson(torch_to_numpy(theta))

    i = 0
    while True:
        i += 1

        # Propose new points in theta space
        theta_prime = theta + sigma * torch.randn(n_walkers, n_dim)

        # Transform to u space
        u_prime, logdetJ_prime = flow.inverse(theta_prime)
        J_flow_prime = logdetJ_prime.sum(-1)

        # Transform to x space
        x_prime, J_prime = scaler.inverse(torch_to_numpy(u_prime))
        x_prime = scaler.apply_boundary_conditions(x_prime)
        x_prime = numpy_to_torch(x_prime)
        J_prime = numpy_to_torch(J_prime)

        # Compute log-likelihood, log-prior, and log-posterior
        P_prime = numpy_to_torch(log_prior(torch_to_numpy(x_prime)))
        finite_prior_mask = torch.isfinite(P_prime)
        L_prime = torch.full((len(x_prime),), -torch.inf)
        L_prime[finite_prior_mask] = numpy_to_torch(log_like(torch_to_numpy(x_prime[finite_prior_mask])))
        n_calls += sum(finite_prior_mask).item()
        Z_prime = numpy_to_torch(log_prob(torch_to_numpy(L_prime), torch_to_numpy(P_prime), beta))

        # Compute Metropolis factors
        alpha = torch.minimum(
            torch.ones(n_walkers),
            torch.exp(Z_prime - Z + J_prime - J + J_flow_prime - J_flow)
        )
        alpha[torch.isnan(alpha)] = 0.0

        # Metropolis criterion
        mask = torch.rand(n_walkers) < alpha

        # Accept new points
        theta[mask] = theta_prime[mask]
        u[mask] = u_prime[mask]
        x[mask] = x_prime[mask]
        J[mask] = J_prime[mask]
        J_flow[mask] = J_flow_prime[mask]
        Z[mask] = Z_prime[mask]
        L[mask] = L_prime[mask]
        P[mask] = P_prime[mask]

        # Adapt scale parameter using diminishing adaptation
        sigma_prime = sigma + 1 / (i + 1) * (torch.mean(alpha) - 0.234)
        if sigma_prime > 1e-4:
            sigma = sigma_prime

        # Compute correlations
        cc_prime = corr.get(torch_to_numpy(theta))

        # Update progress bar if available
        if progress_bar is not None:
            progress_bar.update_stats(
                dict(
                    calls=progress_bar.info['calls'] + sum(finite_prior_mask).item(),
                    accept=torch.mean(alpha).item(),
                    N=i,
                    scale=sigma.item() / (2.38 / np.sqrt(n_dim)),
                    corr=np.mean(cc_prime)
                )
            )

        # Loop termination criteria:
        if corr_threshold is None and i >= int(n_min * ((2.38 / np.sqrt(n_dim)) / sigma.item()) ** 2):
            break
        elif np.mean(cc_prime) < corr_threshold and i >= n_min:
            break
        elif i >= n_max:
            break

    return dict(
        u=torch_to_numpy(u),
        x=torch_to_numpy(x),
        J=torch_to_numpy(J),
        L=torch_to_numpy(L),
        P=torch_to_numpy(P),
        scale=sigma.item(),
        accept=torch.mean(alpha).item(),
        steps=i,
        calls=n_calls
    )


def metropolis(state_dict: dict,
               function_dict: dict,
               option_dict: dict):
    """
    Random-walk Metropolis
    Function that samples the current target distribution
    using a simple Random-walk Metropolis algorithm (i.e. 
    Metropolis-Hastings with Normal proposal distribution).
    
    Parameters
    ----------
    state_dict : dict
        Dictionary of current state
    function_dict : dict
        Dictionary of functions.
    option_dict : dict
        Dictionary of options.
    
    Returns
    -------
    Results dictionary
    """
    # Likelihood call counter
    n_calls = 0

    # Clone state variables
    u = state_dict.get('u').copy()
    x = state_dict.get('x').copy()
    J = state_dict.get('J').copy()
    L = state_dict.get('L').copy()
    P = state_dict.get('P').copy()
    beta = state_dict.get('beta')
    Z = log_prob(L, P, beta)

    # Get functions
    log_like = function_dict.get('loglike')
    log_prior = function_dict.get('logprior')
    scaler = function_dict.get('scaler')

    # Get MCMC options
    n_min = option_dict.get('nmin')
    n_max = option_dict.get('nmax')
    sigma = option_dict.get('sigma')
    corr_threshold = option_dict.get('corr_threshold')
    progress_bar = option_dict.get('progress_bar')

    # Get number of particles and parameters/dimensions
    n_walkers, n_dim = x.shape

    # Compute proposal sample covariance and lower triangular Cholesky in u-space
    cov = np.cov(u.T)
    L_triangular = np.linalg.cholesky(cov)

    # Initialise Pearson correlation object
    corr = Pearson(u)

    i = 0
    while True:
        i += 1

        # Propose new points in u space
        u_prime = u + sigma * np.dot(L_triangular, np.random.randn(n_walkers, n_dim).T).T

        # Transform to x space
        x_prime, J_prime = scaler.inverse(u_prime)
        x_prime = scaler.apply_boundary_conditions(x_prime)

        # Compute log-likelihood, log-prior, and log-posterior
        P_prime = log_prior(x_prime)
        finite_prior_mask = np.isfinite(P_prime)
        L_prime = np.full((len(x_prime),), -np.inf)
        L_prime[finite_prior_mask] = log_like(x_prime[finite_prior_mask])
        Z_prime = log_prob(L_prime, P_prime, beta)
        n_calls += sum(finite_prior_mask)

        # Compute Metropolis factor
        alpha = np.minimum(
            np.ones(len(u_prime)),
            np.exp(Z_prime - Z + J_prime - J)
        )
        alpha[np.isnan(alpha)] = 0.0

        # Metropolis criterion
        mask = np.random.rand(n_walkers) < alpha

        # Accept new points
        u[mask] = u_prime[mask]
        x[mask] = x_prime[mask]
        J[mask] = J_prime[mask]
        Z[mask] = Z_prime[mask]
        L[mask] = L_prime[mask]
        P[mask] = P_prime[mask]

        # Adapt scale parameter using diminishing adaptation
        sigma_prime = sigma + 1 / (i + 1) * (np.mean(alpha) - 0.234)
        if sigma_prime > 1e-4:
            sigma = sigma_prime

        # Compute correlation coefficient
        cc_prime = corr.get(u)

        # Update progress bar if available
        if progress_bar is not None:
            progress_bar.update_stats(
                dict(
                    calls=progress_bar.info['calls'] + sum(finite_prior_mask),
                    accept=np.mean(alpha),
                    N=i,
                    scale=sigma / (2.38 / np.sqrt(n_dim)),
                    corr=np.mean(cc_prime),
                )
            )

        # Termination criteria:
        if corr_threshold is None and i >= int(n_min * ((2.38 / np.sqrt(n_dim)) / sigma) ** 2):
            break
        elif np.mean(cc_prime) < corr_threshold and i >= n_min:
            break
        elif i >= n_max:
            break

    return dict(
        u=u,
        x=x,
        J=J,
        L=L,
        P=P,
        scale=sigma,
        accept=np.mean(alpha),
        steps=i,
        calls=n_calls
    )


def log_prob(L, P, beta):
    """
    Helper function that computes tempered log posterior.
    
    Parameters
    ----------
    L : ``np.ndarray``
        Log-likelihood array
    P : ``np.ndarray``
        Log-prior array
    beta : ``float``
        Beta value
    
    Returns
    -------
    Log-posterior array
    """

    L[np.isnan(L)] = -np.inf
    L[np.isnan(P)] = -np.inf
    L[~np.isfinite(P)] = -np.inf
    P[np.isnan(P)] = -np.inf

    return P + beta * L
