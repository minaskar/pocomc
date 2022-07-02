import numpy as np
import torch

from .tools import numpy_to_torch, torch_to_numpy

class Pearson:
    """
        Pearson correlation coefficient.

    Parameters
    ----------
    a : array of shape (nparticles, ndim)
        Initial positions of particles
    """
    def __init__(self, a):
        self.l = a.shape[0]
        self.am = a - np.sum(a, axis=0) / self.l
        self.aa = np.sum(self.am**2, axis=0) ** 0.5

    def get(self, b):
        """
            Method that computes the correlation coefficients between current positions and initial.

        Parameters
        ----------
        b : array of shape (nparticles, ndim)
            Current positions of particles
        Returns
        -------
        correlation coefficients
        """
        bm = b - np.sum(b, axis=0) / self.l
        bb = np.sum(bm**2, axis=0) ** 0.5
        ab = np.sum(self.am * bm, axis=0)
        return np.abs(ab / (self.aa * bb))


@torch.no_grad()
def PreconditionedMetropolis(state_dict=None,
                             function_dict=None,
                             option_dict=None):
    """
        Preconditioned Metropolis
    
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

    # Clone state variables
    u = torch.clone(numpy_to_torch(state_dict.get('u')))
    x = torch.clone(numpy_to_torch(state_dict.get('x')))
    J = torch.clone(numpy_to_torch(state_dict.get('J')))
    L = torch.clone(numpy_to_torch(state_dict.get('L')))
    P = torch.clone(numpy_to_torch(state_dict.get('P')))
    beta = state_dict.get('beta')
    Z = numpy_to_torch(logprob(torch_to_numpy(L), torch_to_numpy(P), beta))

    # Get functions
    loglike = function_dict.get('loglike')
    logprior = function_dict.get('logprior')
    scaler = function_dict.get('scaler')
    flow = function_dict.get('flow')
    
    # Get MCMC options
    nmin = option_dict.get('nmin')
    nmax = option_dict.get('nmax')
    sigma = option_dict.get('sigma')
    corr_threshold = option_dict.get('corr_threshold')
    progress_bar = option_dict.get('progress_bar')

    # Get number of particles and parameters/dimensions
    nwalkers, ndim = x.shape    

    # Transform u to theta
    theta, logdetJ = flow.forward(u)
    J_flow = -logdetJ.sum(-1)

    # Initialise Pearson correlation object
    corr = Pearson(torch_to_numpy(theta))

    i = 0
    while True:
        i += 1

        # Propose new points in theta space
        theta_prime = theta + sigma * torch.randn(nwalkers, ndim)

        # Transform to u space
        u_prime, logdetJ_prime = flow.inverse(theta_prime)
        J_flow_prime = logdetJ_prime.sum(-1)

        # Transform to x space
        x_prime, J_prime = scaler.inverse(torch_to_numpy(u_prime))
        x_prime = scaler.apply_boundary_conditions(x_prime)
        x_prime = numpy_to_torch(x_prime)
        J_prime = numpy_to_torch(J_prime)

        # Compute log-likelihood, log-prior, and log-posterior
        L_prime = numpy_to_torch(loglike(torch_to_numpy(x_prime)))
        P_prime = numpy_to_torch(logprior(torch_to_numpy(x_prime)))
        Z_prime = numpy_to_torch(logprob(torch_to_numpy(L_prime), torch_to_numpy(P_prime), beta))
        
        # Compute Metropolis factors
        alpha = torch.minimum(torch.ones(nwalkers),
                              torch.exp( Z_prime - Z + J_prime - J + J_flow_prime - J_flow))
        alpha[torch.isnan(alpha)] = 0.0

        # Metropolis criterion
        mask = torch.rand(nwalkers) < alpha

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
        sigma_prime = sigma + 1/(i+1) * (torch.mean(alpha) - 0.234)
        if sigma_prime > 1e-4:
            sigma = sigma_prime

        # Compute correlations
        cc_prime = corr.get(torch_to_numpy(theta))

        # Update progress bar if available
        if progress_bar is not None:
            progress_bar.update_stats(dict(calls=progress_bar.info['calls']+nwalkers,
                                           accept=torch.mean(alpha).item(),
                                           N=i,
                                           scale=sigma.item()/(2.38/np.sqrt(ndim)),
                                           corr=np.mean(cc_prime)))

        # Loop termination criteria:
        if corr_threshold is None and i >= int(nmin * ((2.38/np.sqrt(ndim))/sigma.item())**2):
            break
        elif np.mean(cc_prime) < corr_threshold and i >= nmin:
            break
        elif i >= nmax:
            break

    return dict(u=torch_to_numpy(u),
                x=torch_to_numpy(x),
                J=torch_to_numpy(J),
                L=torch_to_numpy(L),
                P=torch_to_numpy(P),
                scale=sigma.item(),
                accept=torch.mean(alpha).item(),
                steps=i)


def Metropolis(state_dict=None,
               function_dict=None,
               option_dict=None):
    """
        Random-walk Metropolis
    
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

    # Clone state variables
    u = state_dict.get('u').copy()
    x = state_dict.get('x').copy()
    J = state_dict.get('J').copy()
    L = state_dict.get('L').copy()
    P = state_dict.get('P').copy()
    beta = state_dict.get('beta')
    Z = logprob(L, P, beta)

    # Get functions
    loglike = function_dict.get('loglike')
    logprior = function_dict.get('logprior')
    scaler = function_dict.get('scaler')
    
    # Get MCMC options
    nmin = option_dict.get('nmin')
    nmax = option_dict.get('nmax')
    sigma = option_dict.get('sigma')
    corr_threshold = option_dict.get('corr_threshold')
    progress_bar = option_dict.get('progress_bar')

    # Get number of particles and parameters/dimensions
    nwalkers, ndim = x.shape

    # Compute proposal sample covariance and lower triangular Cholesky in u-space
    cov = np.cov(u.T)
    L_triangular = np.linalg.cholesky(cov)

    # Initialise Pearson correlation object
    corr = Pearson(u)

    i = 0
    while True:
        i += 1

        # Propose new points in u space
        u_prime = u + sigma * np.dot(L_triangular, np.random.randn(nwalkers, ndim).T).T

        # Transform to x space
        x_prime, J_prime = scaler.inverse(u_prime)
        x_prime = scaler.apply_boundary_conditions(x_prime)

        # Compute log-likelihood, log-prior, and log-posterior
        L_prime = loglike(x_prime)
        P_prime = logprior(x_prime)
        Z_prime = logprob(L_prime, P_prime, beta)

        # Compute Metropolis factor
        alpha = np.minimum(np.ones(len(u_prime)), np.exp( Z_prime - Z + J_prime - J ))
        alpha[np.isnan(alpha)] = 0.0

        # Metropolis criterion
        mask = np.random.rand(nwalkers) < alpha

        # Accept new points
        u[mask] = u_prime[mask]
        x[mask] = x_prime[mask]
        J[mask] = J_prime[mask]
        Z[mask] = Z_prime[mask]
        L[mask] = L_prime[mask]
        P[mask] = P_prime[mask]

        # Adapt scale parameter using diminishing adaptation
        sigma_prime = sigma + 1/(i+1) * (np.mean(alpha) - 0.234)
        if sigma_prime > 1e-4:
            sigma = sigma_prime

        # Compute correlation coefficient
        cc_prime = corr.get(u)

        # Update progress bar if available
        if progress_bar is not None:
            progress_bar.update_stats(dict(calls=progress_bar.info['calls']+len(u),
                                           accept=np.mean(alpha),
                                           N=i,
                                           scale=sigma/(2.38/np.sqrt(ndim)),
                                           corr=np.mean(cc_prime),

            ))

        # Termination criteria:
        if corr_threshold is None and i >= int(nmin * ((2.38/np.sqrt(ndim))/sigma)**2):
            break
        elif np.mean(cc_prime) < corr_threshold and i >= nmin:
            break
        elif i >= nmax:
            break

    return dict(u=u,
                x=x,
                J=J,
                L=L,
                P=P,
                scale=sigma,
                accept=np.mean(alpha),
                steps=i)


def logprob(L, P, beta):
    """
        Helper function that computes tempered log posterior.
    
    Parameters
    ----------
    L : array
        Log-likelihood array
    P : array
        Log-prior array
    beta : float
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
