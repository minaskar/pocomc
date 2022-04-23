import numpy as np
import torch

from .tools import numpy_to_torch, torch_to_numpy


class Pearson:
    def __init__(self, a):
        self.l = a.shape[0]
        self.am = a - np.sum(a, axis=0) / self.l
        self.aa = np.sum(self.am**2, axis=0) ** 0.5

    def get(self, b):
        bm = b - np.sum(b, axis=0) / self.l
        bb = np.sum(bm**2, axis=0) ** 0.5
        ab = np.sum(self.am * bm, axis=0)
        return np.abs(ab / (self.aa * bb))


@torch.no_grad()
def PreconditionedMetropolis(state_dict=None,
                             function_dict=None,
                             option_dict=None):

    # Get state variables
    u = numpy_to_torch(state_dict.get('u'))
    #x = numpy_to_torch(state_dict.get('x'))
    #J = numpy_to_torch(state_dict.get('J'))
    L = numpy_to_torch(state_dict.get('L'))
    P = numpy_to_torch(state_dict.get('P'))
    beta = state_dict.get('beta')

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

    x, J = scaler.inverse(torch_to_numpy(u))
    x = numpy_to_torch(x)
    J = numpy_to_torch(J)

    nwalkers, ndim = x.shape

    samples = []
    

    if progress_bar is not None:
        progress_bar.update_stats(dict(calls=progress_bar.info['calls']+nwalkers))


    theta, logdetJ = flow.forward(u)
    
    J += -logdetJ.sum(-1)

    Z = numpy_to_torch(logprob(torch_to_numpy(L), torch_to_numpy(P), beta))

    corr = Pearson(torch_to_numpy(theta))

    i = 0
    while True:
        
        theta_prime = theta + sigma * torch.randn(nwalkers, ndim)
        u_prime, logdetJ_prime = flow.inverse(theta_prime)
        x_prime, J_prime = scaler.inverse(torch_to_numpy(u_prime))
        x_prime = numpy_to_torch(x_prime)
        J_prime = numpy_to_torch(J_prime)

        L_prime = numpy_to_torch(loglike(torch_to_numpy(x_prime)))
        P_prime = numpy_to_torch(logprior(torch_to_numpy(x_prime)))
        Z_prime = numpy_to_torch(logprob(torch_to_numpy(L_prime), torch_to_numpy(P_prime), beta))
        J_prime += logdetJ_prime.sum(-1)

        alpha = torch.minimum(torch.ones(len(x_prime)), torch.exp( Z_prime - Z + J_prime - J ))
        alpha[torch.isnan(alpha)] = 0.0
        mask = torch.rand(nwalkers) < alpha

        theta[mask] = theta_prime[mask]
        u[mask] = u_prime[mask]
        x[mask] = x_prime[mask]
        J[mask] = J_prime[mask]
        Z[mask] = Z_prime[mask]
        L[mask] = L_prime[mask]
        P[mask] = P_prime[mask]

        sigma_prime = sigma + 1/(i+1) * (torch.mean(alpha) - 0.234)
        if sigma_prime > 1e-4:
            sigma = sigma_prime

        samples.append(x)
        i += 1

        cc_prime = corr.get(torch_to_numpy(theta))

        if progress_bar is not None:
            progress_bar.update_stats(dict(calls=progress_bar.info['calls']+nwalkers,
                                           accept=torch.mean(alpha).item(),
                                           N=i,
                                           scale=sigma.item()/(2.38/np.sqrt(ndim)),
                                           corr=np.mean(cc_prime),
            ))


        if corr_threshold is None:
            if i >= int(nmin * ((2.38/np.sqrt(ndim))/sigma.item())**2):
                break
        elif np.mean(cc_prime) < corr_threshold and i >= nmin:
            break

        if i >= nmax:
            break

    return dict(u=torch_to_numpy(u),
                x=torch_to_numpy(x),
                J=torch_to_numpy(J),
                Z=torch_to_numpy(Z),
                L=torch_to_numpy(L),
                P=torch_to_numpy(P),
                scale=sigma.item(),
                samples=torch_to_numpy(torch.vstack(samples)),
                accept=torch.mean(alpha).item(),
                steps=i)


def Metropolis(state_dict=None,
               function_dict=None,
               option_dict=None):

    # Get state variables
    u = np.copy(state_dict.get('u'))
    #x = state_dict.get('x')
    #J = state_dict.get('J')
    #L = state_dict.get('L')
    #P = state_dict.get('P')
    beta = state_dict.get('beta')

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

    x, J = scaler.inverse(u) # TODO: remove these
    L = loglike(x)
    P = logprior(x)

    nwalkers, ndim = x.shape

    # Compute proposal sample covariance in u-space
    cov = np.cov(u.T)
    L_triangular = np.linalg.cholesky(cov)

    samples = []
    
    Z = logprob(L, P, beta)

    if progress_bar is not None:
        progress_bar.update_stats(dict(calls=progress_bar.info['calls']))

    corr = Pearson(u)

    i = 0
    while True:

        u_prime = u + sigma * np.dot(L_triangular, np.random.randn(nwalkers, ndim).T).T
        x_prime, J_prime = scaler.inverse(u_prime)
        L_prime = loglike(x_prime)
        P_prime = logprior(x_prime)
        Z_prime = logprob(L, P, beta)

        alpha = np.minimum(np.ones(len(u_prime)), np.exp( Z_prime - Z + J_prime - J ))

        alpha[np.isnan(alpha)] = 0.0
        mask = np.random.rand(nwalkers) < alpha

        u[mask] = u_prime[mask]
        x[mask] = x_prime[mask]
        J[mask] = J_prime[mask]
        Z[mask] = Z_prime[mask]
        L[mask] = L_prime[mask]
        P[mask] = P_prime[mask]

        sigma_prime = sigma + 1/(i+1) * (np.mean(alpha) - 0.234)
        if sigma_prime > 1e-4:
            sigma = sigma_prime

        samples.append(x)
        i += 1

        cc_prime = corr.get(u)

        if progress_bar is not None:
            progress_bar.update_stats(dict(calls=progress_bar.info['calls']+len(u),
                                           accept=np.mean(alpha),
                                           N=i,
                                           scale=sigma/(2.38/np.sqrt(ndim)),
                                           corr=np.mean(cc_prime),

            ))

        if corr_threshold is None:
            if i >= int(nmin * ((2.38/np.sqrt(ndim))/sigma)**2):
                break
        elif np.mean(cc_prime) < corr_threshold and i >= nmin:
            break

        if i >= nmax:
            break

    return dict(u=u,
                x=x,
                J=J,
                Z=Z,
                L=L,
                P=P,
                scale=sigma,
                samples=np.vstack(samples),
                accept=np.mean(alpha),
                steps=i)


def logprob(L, P, beta):
    L[np.isnan(L)] = -np.inf
    L[np.isnan(P)] = -np.inf
    L[~np.isfinite(P)] = -np.inf
    P[np.isnan(P)] = -np.inf

    return P + beta * L
