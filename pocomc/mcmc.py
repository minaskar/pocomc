import torch
from .tools import numpy_to_torch, torch_to_numpy

import numpy as np


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
def PreconditionedMetropolis(logprob,
                             flow,
                             x,
                             nmin,
                             nmax,
                             sigma,
                             target=0.234,
                             adapt=True,
                             corr_threshold=0.9,
                             progress_bar=None,
                             use_maf=False):

    nwalkers, ndim = x.shape

    samples = []
    
    X = torch.clone(numpy_to_torch(x))
    Z, Zl, Zp, J = logprob(torch_to_numpy(X), return_torch=True)

    if progress_bar is not None:
        progress_bar.update_stats(dict(calls=progress_bar.info['calls']+nwalkers))


    if use_maf:
        u, logdetJ = flow.forward(X)
    else:
        u, logdetJ = flow.forward_with_logj(X)

    if use_maf:
        J += -logdetJ.sum(-1)
    else:
        J += -logdetJ

    corr = Pearson(torch_to_numpy(X))

    i = 0
    while True:
        
        u_prime = u + sigma * torch.randn(nwalkers, ndim)
        if use_maf:
            X_prime, logdetJ_prime = flow.inverse(u_prime)
        else:
            X_prime, logdetJ_prime = flow.inverse_with_logj(u_prime)

        Z_prime, Zl_prime, Zp_prime, J_prime = logprob(torch_to_numpy(X_prime), return_torch=True)
        if use_maf:
            J_prime += logdetJ_prime.sum(-1)
        else:
            J_prime += logdetJ_prime

        alpha = torch.minimum(torch.ones(len(X_prime)), torch.exp( Z_prime - Z + J_prime - J ))
        alpha[torch.isnan(alpha)] = 0.0
        mask = torch.rand(nwalkers) < alpha

        u[mask] = u_prime[mask]
        J[mask] = J_prime[mask]
        X[mask] = X_prime[mask]
        Z[mask] = Z_prime[mask]
        Zl[mask] = Zl_prime[mask]
        Zp[mask] = Zp_prime[mask]

        if adapt:
            sigma_prime = sigma + 1/(i+1) * (torch.mean(alpha) - target)
            if sigma_prime > 1e-4:
                sigma = sigma_prime

        samples.append(X)
        i += 1

        cc_prime = corr.get(torch_to_numpy(X_prime))

        if progress_bar is not None:
            progress_bar.update_stats(dict(calls=progress_bar.info['calls']+nwalkers,
                                           accept=torch.mean(alpha).item(),
                                           N=i,
                                           scale=sigma.item()/(2.38/np.sqrt(ndim)),
                                           corr=np.mean(cc_prime),
            ))

        if (np.mean(cc_prime) < corr_threshold and i >= nmin) or i>=nmax:
            break

    return dict(X=torch_to_numpy(X),
                Z=torch_to_numpy(Z),
                Zl=torch_to_numpy(Zl),
                Zp=torch_to_numpy(Zp),
                scale=sigma.item(),
                samples=torch_to_numpy(torch.vstack(samples)),
                accept=torch.mean(alpha).item(),
                steps=i)


def Metropolis(logprob,
               x,
               nmin,
               nmax,
               sigma,
               cov=None,
               target=0.234,
               adapt=True,
               corr_threshold=0.9,
               progress_bar=None):

    nwalkers, ndim = x.shape

    if cov is None:
        cov = np.identity(ndim)
    L = np.linalg.cholesky(cov)

    samples = []
    
    X = np.copy(x)
    Z, Zl, Zp, J = logprob(x, return_torch=False)
    if progress_bar is not None:
        progress_bar.update_stats(dict(calls=progress_bar.info['calls']+len(X)))

    corr = Pearson(x)

    i = 0
    while True:

        X_prime = X + sigma * np.dot(L, np.random.randn(nwalkers, ndim).T).T
        Z_prime, Zl_prime, Zp_prime, J_prime = logprob(X_prime, return_torch=False)

        alpha = np.minimum(np.ones(len(X_prime)), np.exp( Z_prime - Z + J_prime - J))
        alpha[np.isnan(alpha)] = 0.0
        mask = np.random.rand(nwalkers) < alpha

        J[mask] = J_prime[mask]
        X[mask] = X_prime[mask]
        Z[mask] = Z_prime[mask]
        Zl[mask] = Zl_prime[mask]
        Zp[mask] = Zp_prime[mask]

        if adapt:
            sigma_prime = sigma + 1/(i+1) * (np.mean(alpha) - target)
            if sigma_prime > 1e-4:
                sigma = sigma_prime

        samples.append(X)
        i += 1

        cc_prime = corr.get(X_prime)

        if progress_bar is not None:
            progress_bar.update_stats(dict(calls=progress_bar.info['calls']+len(X),
                                           accept=np.mean(alpha),
                                           N=i,
                                           scale=sigma/(2.38/np.sqrt(ndim)),
                                           corr=np.mean(cc_prime),

            ))

        if (np.mean(cc_prime) < corr_threshold and i >= nmin) or i>=nmax:
            break

    return dict(X=X,
                Z=Z,
                Zl=Zl,
                Zp=Zp,
                scale=sigma,
                samples=np.vstack(samples),
                accept=np.mean(alpha),
                steps=i)
