import numpy as np
import torch

from .student import fit_mvstud

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
    blobs = state_dict.get('blobs')
    if blobs is None:
        have_blobs = False
    else:
        have_blobs = True

    # Get functions
    log_like = function_dict.get('loglike')
    log_prior = function_dict.get('logprior')
    scaler = function_dict.get('scaler')
    flow = function_dict.get('flow')
    geometry = function_dict.get('theta_geometry')

    # Get MCMC options
    n_max = option_dict.get('n_max')
    n_steps = option_dict.get('n_steps')
    progress_bar = option_dict.get('progress_bar')
    sigma = np.minimum(option_dict.get('proposal_scale'), 0.99)

    # Get number of particles and parameters/dimensions
    n_walkers, n_dim = x.shape

    # PyTorch variables
    u_t = torch.tensor(u, dtype=torch.float32)
    theta_t, logdetj_flow_t = flow.forward(u_t)
    logdetj_flow_t = -logdetj_flow_t

    mu_t = torch.tensor(geometry.t_mean, dtype=torch.float32)
    cov_t = torch.tensor(geometry.t_cov, dtype=torch.float32)
    nu = geometry.t_nu

    inv_cov_t = torch.linalg.inv(cov_t)
    chol_cov_t = torch.linalg.cholesky(cov_t)

    logp2_val = np.mean(logl + logp)
    cnt = 0

    i = 0
    while True:
        i += 1

        diff_t = theta_t - mu_t
        quad_form_t = torch.einsum('ki,ij,kj->k', diff_t, inv_cov_t, diff_t)
        scale_gamma_t = 2.0 / (nu + quad_form_t)

        gamma_dist = torch.distributions.Gamma((n_dim + nu) / 2, 1.0 / scale_gamma_t)
        s_t = 1.0 / gamma_dist.sample()

        randn_term_t = torch.randn(n_walkers, n_dim) @ chol_cov_t.T
        theta_prime_t = mu_t + (1.0 - sigma ** 2.0) ** 0.5 * diff_t + sigma * torch.sqrt(s_t)[:, None] * randn_term_t

        u_prime_t, logdetj_flow_prime_t = flow.inverse(theta_prime_t)

        # Convert to numpy for scaler and likelihood
        u_prime = u_prime_t.numpy().astype(np.float64)

        # Transform to x space
        x_prime, logdetj_prime = scaler.inverse(u_prime)

        # Apply boundary conditions
        if (scaler.periodic is not None) or (scaler.reflective is not None):
            x_prime = scaler.apply_boundary_conditions_x(x_prime)
            u_prime = scaler.forward(x_prime, check_input=False)
            x_prime, logdetj_prime = scaler.inverse(u_prime)
            u_prime_t = torch.tensor(u_prime, dtype=torch.float32)

        # Compute finite mask
        finite_mask_logdetj_prime = np.isfinite(logdetj_prime)
        finite_mask_x_prime = np.isfinite(x_prime).all(axis=1)
        finite_mask = finite_mask_logdetj_prime & finite_mask_x_prime

        # Evaluate prior
        logp_prime = np.empty(n_walkers)
        logp_prime[finite_mask] = log_prior(x_prime[finite_mask])
        logp_prime[~finite_mask] = -np.inf
        finite_mask_logp = np.isfinite(logp_prime)
        finite_mask = finite_mask & finite_mask_logp
        
        # Evaluate likelihood
        logl_prime = np.empty(n_walkers)
        if have_blobs:
            blobs_prime = np.empty(n_walkers, dtype=np.dtype((blobs[0].dtype, blobs[0].shape)))
            logl_prime[finite_mask], blobs_prime[finite_mask] = log_like(x_prime[finite_mask])
        else:
            logl_prime[finite_mask], _ = log_like(x_prime[finite_mask])
        logl_prime[~finite_mask] = -np.inf
        
        # Update likelihood call counter
        n_calls += np.sum(finite_mask)

        # Compute Metropolis factors
        diff_prime_t = theta_prime_t - mu_t
        quad_form_prime_t = torch.einsum('ki,ij,kj->k', diff_prime_t, inv_cov_t, diff_prime_t)
        A = -(n_dim + nu) / 2 * np.log(1 + quad_form_prime_t.numpy() / nu)
        B = -(n_dim + nu) / 2 * np.log(1 + quad_form_t.numpy() / nu)

        logdetj_flow_prime = logdetj_flow_prime_t.numpy()
        logdetj_flow = logdetj_flow_t.numpy()

        alpha = np.minimum(
            np.ones(n_walkers),
            np.exp(logl_prime * beta - logl * beta + logp_prime - logp + logdetj_prime - logdetj + logdetj_flow_prime - logdetj_flow - A + B)
        )
        alpha[np.isnan(alpha)] = 0.0

        # Metropolis criterion
        u_rand = np.random.rand(n_walkers)
        mask = u_rand < alpha

        # Accept new points
        mask_t = torch.from_numpy(mask)
        theta_t[mask_t] = theta_prime_t[mask_t]
        u_t[mask_t] = u_prime_t[mask_t]
        logdetj_flow_t[mask_t] = logdetj_flow_prime_t[mask_t]
        u[mask] = u_prime[mask]
        x[mask] = x_prime[mask]
        logdetj[mask] = logdetj_prime[mask]
        logl[mask] = logl_prime[mask]
        logp[mask] = logp_prime[mask]
        if have_blobs:
            blobs[mask] = blobs_prime[mask]

        # Adapt scale parameter using diminishing adaptation
        sigma = np.abs(np.minimum(sigma + 1 / (i + 1)**0.75 * (np.mean(alpha) - 0.234), np.minimum(2.38 / n_dim**0.5, 0.99)))
        #sigma = np.minimum(sigma + 1 / (i + 1)**0.5 * (np.mean(alpha) - 0.234), 0.99)

        # Adapt mean parameter using diminishing adaptation
        mu_t = mu_t + 1.0 / (i + 1.0) * (torch.mean(theta_t, axis=0) - mu_t)

        # Update progress bar if available
        if progress_bar is not None:
            progress_bar.update_stats(
                dict(calls=progress_bar.info['calls'] + np.sum(finite_mask),
                    acc=np.mean(alpha),
                    steps=i,
                    logP=np.mean(logl + logp),
                    eff=sigma / (2.38 / np.sqrt(n_dim)),
                    )
            )

        # Loop termination criteria:
        logp2_val_new = np.mean(logl + logp)
        if logp2_val_new > logp2_val:
            cnt = 0
            logp2_val = logp2_val_new
        else:
            cnt += 1
            if cnt >= n_steps * ((2.38 / n_dim**0.5) / sigma)**2.0:
                break

        if i >= n_max:
            break

    return dict(u=u, x=x, logdetj=logdetj, logl=logl, logp=logp, blobs=blobs, efficiency=sigma, 
                accept=np.mean(alpha), steps=i, calls=n_calls, proposal_scale=sigma)

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
    blobs = state_dict.get('blobs')
    if blobs is None:
        have_blobs = False
    else:
        have_blobs = True

    # Get functions
    log_like = function_dict.get('loglike')
    log_prior = function_dict.get('logprior')
    scaler = function_dict.get('scaler')
    flow = function_dict.get('flow')
    geometry = function_dict.get('theta_geometry')

    # Get MCMC options
    n_max = option_dict.get('n_max')
    n_steps = option_dict.get('n_steps')
    progress_bar = option_dict.get('progress_bar')
    sigma = option_dict.get('proposal_scale')

    # Get number of particles and parameters/dimensions
    n_walkers, n_dim = x.shape

    cov_t = torch.tensor(geometry.normal_cov, dtype=torch.float32)
    chol_t = torch.linalg.cholesky(cov_t)

    # PyTorch variables
    u_t = torch.tensor(u, dtype=torch.float32)
    theta_t, logdetj_flow_t = flow.forward(u_t)
    logdetj_flow_t = -logdetj_flow_t

    logp2_val = np.mean(logl + logp + logdetj)
    cnt = 0

    i = 0
    while True:
        i += 1

        # Propose new points in theta space
        randn_term_t = torch.randn(n_walkers, n_dim) @ chol_t.T
        theta_prime_t = theta_t + sigma * randn_term_t

        # Transform to u space
        u_prime_t, logdetj_flow_prime_t = flow.inverse(theta_prime_t)

        u_prime = u_prime_t.numpy().astype(np.float64)

        # Transform to x space
        x_prime, logdetj_prime = scaler.inverse(u_prime)

        # Apply boundary conditions
        if (scaler.periodic is not None) or (scaler.reflective is not None):
            x_prime = scaler.apply_boundary_conditions_x(x_prime)
            u_prime = scaler.forward(x_prime, check_input=False)
            x_prime, logdetj_prime = scaler.inverse(u_prime)
            u_prime_t = torch.tensor(u_prime, dtype=torch.float32)

        # Compute finite mask
        finite_mask_logdetj_prime = np.isfinite(logdetj_prime)
        finite_mask_x_prime = np.isfinite(x_prime).all(axis=1)
        finite_mask = finite_mask_logdetj_prime & finite_mask_x_prime

        # Evaluate prior
        logp_prime = np.empty(n_walkers)
        logp_prime[finite_mask] = log_prior(x_prime[finite_mask])
        logp_prime[~finite_mask] = -np.inf
        finite_mask_logp = np.isfinite(logp_prime)
        finite_mask = finite_mask & finite_mask_logp

        # Compute log-likelihood, log-prior, and log-posterior
        logl_prime = np.empty(n_walkers)
        if have_blobs:
            blobs_prime = np.empty(n_walkers, dtype=np.dtype((blobs[0].dtype, blobs[0].shape)))
            logl_prime[finite_mask], blobs_prime[finite_mask] = log_like(x_prime[finite_mask])
        else:
            logl_prime[finite_mask], _ = log_like(x_prime[finite_mask])
        logl_prime[~finite_mask] = -np.inf

        # Update likelihood call counter
        n_calls += np.sum(finite_mask)

        logdetj_flow_prime = logdetj_flow_prime_t.numpy()
        logdetj_flow = logdetj_flow_t.numpy()

        # Compute Metropolis factors
        alpha = np.minimum(
            np.ones(n_walkers),
            np.exp(logl_prime * beta - logl * beta + logp_prime - logp + logdetj_prime - logdetj + logdetj_flow_prime - logdetj_flow)
        )
        alpha[np.isnan(alpha)] = 0.0

        # Metropolis criterion
        u_rand = np.random.rand(n_walkers)
        mask = u_rand < alpha

        # Accept new points
        mask_t = torch.from_numpy(mask)
        theta_t[mask_t] = theta_prime_t[mask_t]
        u_t[mask_t] = u_prime_t[mask_t]
        logdetj_flow_t[mask_t] = logdetj_flow_prime_t[mask_t]
        u[mask] = u_prime[mask]
        x[mask] = x_prime[mask]
        logdetj[mask] = logdetj_prime[mask]
        logl[mask] = logl_prime[mask]
        logp[mask] = logp_prime[mask]
        if have_blobs:
            blobs[mask] = blobs_prime[mask]

        # Adapt scale parameter using diminishing adaptation
        sigma = sigma + 1 / (i + 1) * (np.mean(alpha) - 0.234)

        # Update progress bar if available
        if progress_bar is not None:
            progress_bar.update_stats(
                dict(calls=progress_bar.info['calls'] + np.sum(finite_mask),
                    acc=np.mean(alpha),
                    steps=i,
                    logP=np.mean(logl + logp),
                    eff=sigma / (2.38 / np.sqrt(n_dim)))
            )

        # Loop termination criteria:
        logp2_val_new = np.mean(logl + logp + logdetj)
        if logp2_val_new > logp2_val:
            cnt = 0
            logp2_val = logp2_val_new
        else:
            cnt += 1
            if cnt >= n_steps * (np.minimum(1.0, (2.38 / n_dim**0.5) / sigma))**2.0:
                break

        if i >= n_max:
            break


    return dict(u=u, x=x, logdetj=logdetj, logl=logl, logp=logp, blobs=blobs, efficiency=sigma, 
                accept=np.mean(alpha), steps=i, calls=n_calls, proposal_scale=sigma)


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
    blobs = state_dict.get('blobs')
    if blobs is None:
        have_blobs = False
    else:
        have_blobs = True

    # Get functions
    log_like = function_dict.get('loglike')
    log_prior = function_dict.get('logprior')
    scaler = function_dict.get('scaler')
    geometry = function_dict.get('u_geometry')

    # Get MCMC options
    n_max = option_dict.get('n_max')
    n_steps = option_dict.get('n_steps')
    progress_bar = option_dict.get('progress_bar')
    sigma = np.minimum(option_dict.get('proposal_scale'), 0.99)

    # Get number of particles and parameters/dimensions
    n_walkers, n_dim = x.shape

    mu = geometry.t_mean
    cov = geometry.t_cov
    nu = geometry.t_nu

    inv_cov = np.linalg.inv(cov)
    chol_cov = np.linalg.cholesky(cov)

    logp2_val = np.mean(logl + logp)
    #logp2_val = np.mean(logl * beta + logp)
    cnt = 0

    i = 0
    while True:
        i += 1

        diff = u - mu
        quad_form = np.einsum('ki,ij,kj->k', diff, inv_cov, diff)
        scale_gamma = 2.0 / (nu + quad_form)
        s = 1.0 / np.random.gamma((n_dim + nu) / 2, scale_gamma, size=n_walkers)

        # Propose new points in u space
        randn_term = np.random.randn(n_walkers, n_dim) @ chol_cov.T
        u_prime = mu + (1.0 - sigma ** 2.0) ** 0.5 * diff + sigma * np.sqrt(s)[:, None] * randn_term

        # Transform to x space
        x_prime, logdetj_prime = scaler.inverse(u_prime)

        # Apply boundary conditions
        if (scaler.periodic is not None) or (scaler.reflective is not None):
            x_prime = scaler.apply_boundary_conditions_x(x_prime)
            u_prime = scaler.forward(x_prime, check_input=False)
            x_prime, logdetj_prime = scaler.inverse(u_prime)

        # Compute finite mask
        finite_mask_logdetj_prime = np.isfinite(logdetj_prime)
        finite_mask_x_prime = np.isfinite(x_prime).all(axis=1)
        finite_mask = finite_mask_logdetj_prime & finite_mask_x_prime

        # Evaluate prior
        logp_prime = np.empty(n_walkers)
        logp_prime[finite_mask] = log_prior(x_prime[finite_mask])
        logp_prime[~finite_mask] = -np.inf
        finite_mask_logp = np.isfinite(logp_prime)
        finite_mask = finite_mask & finite_mask_logp

        # Evaluate likelihood
        logl_prime = np.empty(n_walkers)
        if have_blobs:
            blobs_prime = np.empty(n_walkers, dtype=np.dtype((blobs[0].dtype, blobs[0].shape)))
            logl_prime[finite_mask], blobs_prime[finite_mask] = log_like(x_prime[finite_mask])
        else:
            logl_prime[finite_mask], _ = log_like(x_prime[finite_mask])
        logl_prime[~finite_mask] = -np.inf
        
        # Update likelihood call counter
        n_calls += np.sum(finite_mask)

        # Compute Metropolis factors
        diff_prime = u_prime - mu
        quad_form_prime = np.einsum('ki,ij,kj->k', diff_prime, inv_cov, diff_prime)
        A = -(n_dim + nu) / 2 * np.log(1 + quad_form_prime / nu)
        B = -(n_dim + nu) / 2 * np.log(1 + quad_form / nu)

        alpha = np.minimum(
            np.ones(n_walkers),
            np.exp(logl_prime * beta - logl * beta + logp_prime - logp + logdetj_prime - logdetj - A + B)
        )
        alpha[np.isnan(alpha)] = 0.0

        # Metropolis criterion
        u_rand = np.random.rand(n_walkers)
        mask = u_rand < alpha

        # Accept new points
        u[mask] = u_prime[mask]
        x[mask] = x_prime[mask]
        logdetj[mask] = logdetj_prime[mask]
        logl[mask] = logl_prime[mask]
        logp[mask] = logp_prime[mask]
        if have_blobs:
            blobs[mask] = blobs_prime[mask]

        # Adapt scale parameter using diminishing adaptation
        sigma = np.abs(np.minimum(sigma + 1 / (i + 1)**0.75 * (np.mean(alpha) - 0.234), np.minimum(2.38 / n_dim**0.5, 0.99)))
        #sigma = sigma + 1 / (i + 1)**0.75 * (np.mean(alpha) - 0.234)

        # Update progress bar if available
        if progress_bar is not None:
            progress_bar.update_stats(
                dict(calls=progress_bar.info['calls'] + np.sum(finite_mask),
                    acc=np.mean(alpha),
                    steps=i,
                    logP=np.mean(logl + logp),
                    eff=sigma / (2.38 / np.sqrt(n_dim)))
            )

        # Loop termination criteria:
        logp2_val_new = np.mean(logl + logp)
        if logp2_val_new > logp2_val:
            cnt = 0
            logp2_val = logp2_val_new
        else:
            cnt += 1
            if cnt >= n_steps * ((2.38 / n_dim**0.5) / sigma)**2.0:
                break

        if i >= n_max:
            break

    return dict(u=u, x=x, logdetj=logdetj, logl=logl, logp=logp, blobs=blobs, efficiency=sigma, 
                accept=np.mean(alpha), steps=i, calls=n_calls, proposal_scale=sigma)

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
    blobs = state_dict.get('blobs')
    if blobs is None:
        have_blobs = False
    else:
        have_blobs = True

    # Get functions
    log_like = function_dict.get('loglike')
    log_prior = function_dict.get('logprior')
    scaler = function_dict.get('scaler')
    geometry = function_dict.get('u_geometry')

    # Get MCMC options
    n_max = option_dict.get('n_max')
    n_steps = option_dict.get('n_steps')
    progress_bar = option_dict.get('progress_bar')
    sigma = option_dict.get('proposal_scale')

    # Get number of particles and parameters/dimensions
    n_walkers, n_dim = x.shape

    cov = geometry.normal_cov
    chol = np.linalg.cholesky(cov)

    logp2_val = np.mean(logl + logp + logdetj)
    cnt = 0

    i = 0
    while True:
        i += 1

        # Propose new points in theta space
        randn_term = np.random.randn(n_walkers, n_dim) @ chol.T
        u_prime = u + sigma * randn_term

        # Transform to x space
        x_prime, logdetj_prime = scaler.inverse(u_prime)

        # Apply boundary conditions
        if (scaler.periodic is not None) or (scaler.reflective is not None):
            x_prime = scaler.apply_boundary_conditions_x(x_prime)
            u_prime = scaler.forward(x_prime, check_input=False)
            x_prime, logdetj_prime = scaler.inverse(u_prime)

        # Compute finite mask
        finite_mask_logdetj_prime = np.isfinite(logdetj_prime)
        finite_mask_x_prime = np.isfinite(x_prime).all(axis=1)
        finite_mask = finite_mask_logdetj_prime & finite_mask_x_prime

        # Evaluate prior
        logp_prime = np.empty(n_walkers)
        logp_prime[finite_mask] = log_prior(x_prime[finite_mask])
        logp_prime[~finite_mask] = -np.inf
        finite_mask_logp = np.isfinite(logp_prime)
        finite_mask = finite_mask & finite_mask_logp

        # Evaluate likelihood
        logl_prime = np.empty(n_walkers)
        if have_blobs:
            blobs_prime = np.empty(n_walkers, dtype=np.dtype((blobs[0].dtype, blobs[0].shape)))
            logl_prime[finite_mask], blobs_prime[finite_mask] = log_like(x_prime[finite_mask])
        else:
            logl_prime[finite_mask], _ = log_like(x_prime[finite_mask])
        logl_prime[~finite_mask] = -np.inf

        # Update likelihood call counter
        n_calls += np.sum(finite_mask)

        # Compute Metropolis factors
        alpha = np.minimum(
            np.ones(n_walkers),
            np.exp(logl_prime * beta - logl * beta + logp_prime - logp + logdetj_prime - logdetj)
        )
        alpha[np.isnan(alpha)] = 0.0

        # Metropolis criterion
        u_rand = np.random.rand(n_walkers)
        mask = u_rand < alpha

        # Accept new points
        u[mask] = u_prime[mask]
        x[mask] = x_prime[mask]
        logdetj[mask] = logdetj_prime[mask]
        logl[mask] = logl_prime[mask]
        logp[mask] = logp_prime[mask]
        if have_blobs:
            blobs[mask] = blobs_prime[mask]

        # Adapt scale parameter using diminishing adaptation
        sigma = np.abs(sigma + 1 / (i + 1) * (np.mean(alpha) - 0.234))

        # Update progress bar if available
        if progress_bar is not None:
            progress_bar.update_stats(
                dict(calls=progress_bar.info['calls'] + np.sum(finite_mask),
                    acc=np.mean(alpha),
                    steps=i,
                    logP=np.mean(logl + logp),
                    eff=sigma / (2.38 / np.sqrt(n_dim)))
            )

        # Loop termination criteria:
        logp2_val_new = np.mean(logl + logp + logdetj)
        if logp2_val_new > logp2_val:
            cnt = 0
            logp2_val = logp2_val_new
        else:
            cnt += 1
            if cnt >= n_steps * ((2.38 / n_dim**0.5) / sigma)**2.0:
                break

        if i >= n_max:
            break


    return dict(u=u, x=x, logdetj=logdetj, logl=logl, logp=logp, blobs=blobs, efficiency=sigma, 
                accept=np.mean(alpha), steps=i, calls=n_calls, proposal_scale=sigma)
