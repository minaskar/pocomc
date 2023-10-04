import numpy as np
import torch

from .tools import numpy_to_torch, torch_to_numpy
from .student import fit_mvstud

class flow_numpy_wrapper:
    """
    Wrapper class for numpy flows.

    Parameters
    ----------
    flow : Flow object
        Flow object that implements forward and inverse
        transformations.
    
    Returns
    -------
    Flow object
    """
    def __init__(self, flow):
        self.flow = flow

    @torch.no_grad()
    def forward(self, v):
        v = numpy_to_torch(v)
        theta, logdetj = self.flow.forward(v)
        theta = torch_to_numpy(theta)
        logdetj = - torch_to_numpy(logdetj)
        return theta, logdetj

    @torch.no_grad()
    def inverse(self, theta):
        theta = numpy_to_torch(theta)
        v, logdetj = self.flow.inverse(theta)
        v = torch_to_numpy(v)
        logdetj = torch_to_numpy(logdetj)
        return v, logdetj

@torch.no_grad()
def preconditioned_pcn(state_dict: dict,
                       function_dict: dict,
                       option_dict: dict):
    """
    Doubly Preconditioned Crank-Nicolson
    
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
    u = np.copy(state_dict.get('u'))
    x = np.copy(state_dict.get('x'))
    logdetj = np.copy(state_dict.get('logdetj'))
    logl = np.copy(state_dict.get('logl'))
    logp = np.copy(state_dict.get('logp'))
    beta = state_dict.get('beta')

    # Get functions
    log_like = function_dict.get('loglike')
    log_prior = function_dict.get('logprior')
    scaler = function_dict.get('scaler')
    flow = flow_numpy_wrapper(function_dict.get('flow'))

    # Get MCMC options
    n_max = option_dict.get('nmax')
    progress_bar = option_dict.get('progress_bar')
    patience = option_dict.get('patience')

    # Get number of particles and parameters/dimensions
    n_walkers, n_dim = x.shape

    # Transform u to theta
    theta, logdetj_flow = flow.forward(u)

    sigma = np.minimum(2.38 / n_dim**0.5, 0.99)

    mu, cov, nu = fit_mvstud(theta)
    if ~np.isfinite(nu):
        nu = 1e6

    inv_cov = np.linalg.inv(cov)
    chol_cov = np.linalg.cholesky(cov)

    #logp2_val = np.mean(logl + logp + logdetj)
    logp2_val = np.mean(logl + logp)
    cnt = 0

    i = 0
    while True:
        i += 1

        diff = theta - mu
        s = np.empty(n_walkers)
        for k in range(n_walkers):
            s[k] = 1./np.random.gamma((n_dim + nu) / 2, 2.0/(nu + np.dot(diff[k],np.dot(inv_cov,diff[k]))))

        # Propose new points in theta space
        theta_prime = np.empty((n_walkers, n_dim))
        for k in range(n_walkers):
            theta_prime[k] = mu + (1.0 - sigma ** 2.0) ** 0.5 * diff[k] + sigma * np.sqrt(s[k]) * np.dot(chol_cov, np.random.randn(n_dim))        

        # Transform to u space
        u_prime, logdetj_flow_prime = flow.inverse(theta_prime)

        # Transform to x space
        x_prime, logdetj_prime = scaler.inverse(u_prime)

        # Compute log-likelihood, log-prior, and log-posterior
        u_rand = np.random.rand(n_walkers)

        logl_prime = log_like(x_prime)
        logp_prime = log_prior(x_prime)

        n_calls += len(logl_prime)

        # Compute Metropolis factors
        diff_prime = theta_prime-mu
        A = np.empty(n_walkers)
        B = np.empty(n_walkers)
        for k in range(n_walkers):
            A[k] = -(n_dim+nu)/2*np.log(1+np.dot(diff_prime[k],np.dot(inv_cov,diff_prime[k]))/nu)
            B[k] = -(n_dim+nu)/2*np.log(1+np.dot(diff[k],np.dot(inv_cov,diff[k]))/nu)
        alpha = np.minimum(
            np.ones(n_walkers),
            np.exp(logl_prime * beta - logl * beta + logp_prime - logp + logdetj_prime - logdetj + logdetj_flow_prime - logdetj_flow - A + B)
        )
        alpha[np.isnan(alpha)] = 0.0

        # Metropolis criterion
        mask = u_rand < alpha

        theta_squared_distance = np.sum((theta_prime - theta)**2, axis=1)

        # Accept new points
        theta[mask] = theta_prime[mask]
        u[mask] = u_prime[mask]
        x[mask] = x_prime[mask]
        logdetj[mask] = logdetj_prime[mask]
        logdetj_flow[mask] = logdetj_flow_prime[mask]
        logl[mask] = logl_prime[mask]
        logp[mask] = logp_prime[mask]

        # Adapt scale parameter using diminishing adaptation
        sigma = np.abs(np.minimum(sigma + 1 / (i + 1) * (np.mean(alpha) - 0.234), np.minimum(2.38 / n_dim**0.5, 0.99)))

        # Adapt mean parameter using diminishing adaptation
        mu = mu + 1.0 / (i + 1.0) * (np.mean(theta, axis=0) - mu)

        # Update progress bar if available
        if progress_bar is not None:
            progress_bar.update_stats(
                dict(calls=progress_bar.info['calls'] + n_walkers,
                    accept=np.mean(alpha),
                    steps=i,
                    logp=np.mean(logl + logp + logdetj),
                    efficiency=sigma / (2.38 / np.sqrt(n_dim)),
                    )
            )

        # Loop termination criteria:
        #logp2_val_new = np.mean(logl + logp + logdetj)
        logp2_val_new = np.mean(logl + logp)
        if logp2_val_new > logp2_val:
            cnt = 0
            logp2_val = logp2_val_new
        else:
            cnt += 1
        if patience is None:
            if cnt >= n_dim // 2 * ((2.38 / n_dim**0.5) / sigma)**1.5 * np.minimum(1.0, np.abs(0.234 / np.mean(alpha))):
                break
        else:
            if cnt >= patience:
                break

        if i >= n_max:
            break

    return dict(u=u, x=x, logdetj=logdetj, logl=logl, logp=logp, efficiency=sigma, accept=np.mean(alpha), steps=i, calls=n_calls)

@torch.no_grad()
def preconditioned_rwm(state_dict: dict,
                       function_dict: dict,
                       option_dict: dict):
    """
    Preconditioned Random-walk Metropolis
    
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
    u = np.copy(state_dict.get('u'))
    x = np.copy(state_dict.get('x'))
    logdetj = np.copy(state_dict.get('logdetj'))
    logl = np.copy(state_dict.get('logl'))
    logp = np.copy(state_dict.get('logp'))
    beta = state_dict.get('beta')

    # Get functions
    log_like = function_dict.get('loglike')
    log_prior = function_dict.get('logprior')
    scaler = function_dict.get('scaler')
    flow = flow_numpy_wrapper(function_dict.get('flow'))

    # Get MCMC options
    n_max = option_dict.get('nmax')
    progress_bar = option_dict.get('progress_bar')

    # Get number of particles and parameters/dimensions
    n_walkers, n_dim = x.shape

    sigma = 2.38/n_dim**0.5

    # Transform u to theta
    theta, logdetj_flow = flow.forward(u)

    logp2_val = np.mean(logl + logp + logdetj)
    cnt = 0

    i = 0
    while True:
        i += 1

        # Propose new points in theta space
        theta_prime = theta + sigma * np.random.randn(n_walkers, n_dim)

        # Transform to u space
        u_prime, logdetj_flow_prime = flow.inverse(theta_prime)

        # Transform to x space
        x_prime, logdetj_prime = scaler.inverse(u_prime)

        # Compute log-likelihood, log-prior, and log-posterior
        u_rand = np.random.rand(n_walkers)

        logl_prime = log_like(x_prime)
        logp_prime = log_prior(x_prime)

        n_calls += len(logl_prime)

        # Compute Metropolis factors
        alpha = np.minimum(
            np.ones(n_walkers),
            np.exp(logl_prime * beta - logl * beta + logp_prime - logp + logdetj_prime - logdetj + logdetj_flow_prime - logdetj_flow)
        )
        alpha[np.isnan(alpha)] = 0.0

        # Metropolis criterion
        mask = u_rand < alpha

        # Accept new points
        theta[mask] = theta_prime[mask]
        u[mask] = u_prime[mask]
        x[mask] = x_prime[mask]
        logdetj[mask] = logdetj_prime[mask]
        logdetj_flow[mask] = logdetj_flow_prime[mask]
        logl[mask] = logl_prime[mask]
        logp[mask] = logp_prime[mask]

        # Adapt scale parameter using diminishing adaptation
        sigma = sigma + 1 / (i + 1) * (np.mean(alpha) - 0.234)

        # Update progress bar if available
        if progress_bar is not None:
            progress_bar.update_stats(
                dict(calls=progress_bar.info['calls'] + n_walkers,
                    accept=np.mean(alpha),
                    steps=i,
                    logp=np.mean(logl + logp + logdetj),
                    efficiency=sigma / (2.38 / np.sqrt(n_dim)))
            )

        # Loop termination criteria:
        logp2_val_new = np.mean(logl + logp + logdetj)
        if logp2_val_new > logp2_val:
            cnt = 0
            logp2_val = logp2_val_new
        else:
            cnt += 1
        if cnt >= n_dim // 2 * (np.minimum(1.0, (2.38 / n_dim**0.5) / sigma))**2.0 * np.minimum(1.0, np.abs(0.234 / np.mean(alpha))):
            break

        if i >= n_max:
            break


    return dict(u=u, x=x, logdetj=logdetj, logl=logl, logp=logp, efficiency=sigma, accept=np.mean(alpha), steps=i, calls=n_calls)


def pcn(state_dict: dict,
        function_dict: dict,
        option_dict: dict):
    """
    Preconditioned Crank-Nicolson
    
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
    u = np.copy(state_dict.get('u'))
    x = np.copy(state_dict.get('x'))
    logdetj = np.copy(state_dict.get('logdetj'))
    logl = np.copy(state_dict.get('logl'))
    logp = np.copy(state_dict.get('logp'))
    beta = state_dict.get('beta')

    # Get functions
    log_like = function_dict.get('loglike')
    log_prior = function_dict.get('logprior')
    scaler = function_dict.get('scaler')

    # Get MCMC options
    n_max = option_dict.get('nmax')
    progress_bar = option_dict.get('progress_bar')

    # Get number of particles and parameters/dimensions
    n_walkers, n_dim = x.shape

    #sigma = np.minimum(2.38 / n_dim**0.5, 0.99)
    sigma = 0.5

    mu, cov, nu = fit_mvstud(u)
    if ~np.isfinite(nu):
        nu = 1e6

    inv_cov = np.linalg.inv(cov)
    chol_cov = np.linalg.cholesky(cov)

    logp2_val = np.mean(logl + logp + logdetj)
    cnt = 0

    i = 0
    while True:
        i += 1

        diff = u - mu
        s = np.empty(n_walkers)
        for k in range(n_walkers):
            s[k] = 1./np.random.gamma((n_dim + nu) / 2, 2.0/(nu + np.dot(diff[k],np.dot(inv_cov,diff[k]))))

        # Propose new points in u space
        u_prime = np.empty((n_walkers, n_dim))
        for k in range(n_walkers):
            u_prime[k] = mu + (1.0 - sigma ** 2.0) ** 0.5 * diff[k] + sigma * np.sqrt(s[k]) * np.dot(chol_cov, np.random.randn(n_dim))        

        # Transform to x space
        x_prime, logdetj_prime = scaler.inverse(u_prime)

        # Compute log-likelihood, log-prior, and log-posterior
        u_rand = np.random.rand(n_walkers)

        logl_prime = log_like(x_prime)
        logp_prime = log_prior(x_prime)

        n_calls += len(logl_prime)

        # Compute Metropolis factors
        diff_prime = u_prime - mu
        A = np.empty(n_walkers)
        B = np.empty(n_walkers)
        for k in range(n_walkers):
            A[k] = -(n_dim+nu)/2*np.log(1+np.dot(diff_prime[k],np.dot(inv_cov,diff_prime[k]))/nu)
            B[k] = -(n_dim+nu)/2*np.log(1+np.dot(diff[k],np.dot(inv_cov,diff[k]))/nu)
        alpha = np.minimum(
            np.ones(n_walkers),
            np.exp(logl_prime * beta - logl * beta + logp_prime - logp + logdetj_prime - logdetj - A + B)
        )
        alpha[np.isnan(alpha)] = 0.0

        # Metropolis criterion
        mask = u_rand < alpha

        # Accept new points
        u[mask] = u_prime[mask]
        x[mask] = x_prime[mask]
        logdetj[mask] = logdetj_prime[mask]
        logl[mask] = logl_prime[mask]
        logp[mask] = logp_prime[mask]

        # Adapt scale parameter using diminishing adaptation
        if sigma < 0.5 or np.mean(alpha) < 0.40:
            sigma = np.abs(np.minimum(sigma + 1 / (i + 1) * (np.mean(alpha) - 0.4), np.minimum(2.38 / n_dim**0.5, 0.5)))

        # Update progress bar if available
        if progress_bar is not None:
            progress_bar.update_stats(
                dict(calls=progress_bar.info['calls'] + n_walkers,
                    accept=np.mean(alpha),
                    steps=i,
                    logp=np.mean(logl + logp + logdetj),
                    #efficiency=sigma * np.mean(np.sqrt(s)) / (2.38 / np.sqrt(n_dim)))
                    efficiency=sigma / (2.38 / np.sqrt(n_dim)))
            )

        # Loop termination criteria:
        logp2_val_new = np.mean(logl + logp + logdetj)
        if logp2_val_new > logp2_val:
            cnt = 0
            logp2_val = logp2_val_new
        else:
            cnt += 1
        if cnt >= n_dim // 2 * (np.minimum(1.0, (2.38 / n_dim**0.5) / sigma))**1.5 * np.minimum(1.0, np.abs(0.4 / np.mean(alpha))):
            break

        if i >= n_max:
            break

    return dict(u=u, x=x, logdetj=logdetj, logl=logl, logp=logp, efficiency=sigma, accept=np.mean(alpha), steps=i, calls=n_calls)

def rwm(state_dict: dict,
        function_dict: dict,
        option_dict: dict):
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
    # Likelihood call counter
    n_calls = 0

    # Clone state variables
    u = np.copy(state_dict.get('u'))
    x = np.copy(state_dict.get('x'))
    logdetj = np.copy(state_dict.get('logdetj'))
    logl = np.copy(state_dict.get('logl'))
    logp = np.copy(state_dict.get('logp'))
    beta = state_dict.get('beta')

    # Get functions
    log_like = function_dict.get('loglike')
    log_prior = function_dict.get('logprior')
    scaler = function_dict.get('scaler')

    # Get MCMC options
    n_max = option_dict.get('nmax')
    progress_bar = option_dict.get('progress_bar')

    # Get number of particles and parameters/dimensions
    n_walkers, n_dim = x.shape

    sigma = 0.5 * 2.38/n_dim**0.5

    cov = np.cov(u.T)
    chol = np.linalg.cholesky(cov)

    logp2_val = np.mean(logl + logp + logdetj)
    cnt = 0

    i = 0
    while True:
        i += 1

        # Propose new points in theta space
        u_prime = np.empty((n_walkers, n_dim))
        for k in range(n_walkers):
            u_prime[k] = u[k] + sigma * np.dot(chol, np.random.randn(n_dim))

        # Transform to x space
        x_prime, logdetj_prime = scaler.inverse(u_prime)

        # Compute log-likelihood, log-prior, and log-posterior
        u_rand = np.random.rand(n_walkers)

        logl_prime = log_like(x_prime)
        logp_prime = log_prior(x_prime)

        n_calls += len(logl_prime)

        # Compute Metropolis factors
        alpha = np.minimum(
            np.ones(n_walkers),
            np.exp(logl_prime * beta - logl * beta + logp_prime - logp + logdetj_prime - logdetj)
        )
        alpha[np.isnan(alpha)] = 0.0

        # Metropolis criterion
        mask = u_rand < alpha

        # Accept new points
        u[mask] = u_prime[mask]
        x[mask] = x_prime[mask]
        logdetj[mask] = logdetj_prime[mask]
        logl[mask] = logl_prime[mask]
        logp[mask] = logp_prime[mask]

        # Adapt scale parameter using diminishing adaptation
        sigma = np.abs(sigma + 1 / (i + 1) * (np.mean(alpha) - 0.234))

        # Update progress bar if available
        if progress_bar is not None:
            progress_bar.update_stats(
                dict(calls=progress_bar.info['calls'] + n_walkers,
                    accept=np.mean(alpha),
                    steps=i,
                    logp=np.mean(logl + logp + logdetj),
                    efficiency=sigma / (2.38 / np.sqrt(n_dim)))
            )

        # Loop termination criteria:
        logp2_val_new = np.mean(logl + logp + logdetj)
        if logp2_val_new > logp2_val:
            cnt = 0
            logp2_val = logp2_val_new
        else:
            cnt += 1
        if cnt >= n_dim // 2 * ((2.38 / n_dim**0.5) / sigma)**2.0 * np.minimum(1.0, np.abs(0.234 / np.mean(alpha))):
            break

        if i >= n_max:
            break


    return dict(u=u, x=x, logdetj=logdetj, logl=logl, logp=logp, efficiency=sigma, accept=np.mean(alpha), steps=i, calls=n_calls)
