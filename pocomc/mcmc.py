import torch
from .tools import torch_to_numpy

@torch.no_grad()
def RandomWalkMetropolis(logprob, flow, x, N, sigma, target=0.234, adapt=True):

    nwalkers, ndim = x.shape
    
    X = torch.clone(x)
    Z, Zl, Zp = logprob(torch_to_numpy(x), return_torch=True)

    u, logdetJ = flow.forward(x)
    J = -logdetJ.sum(-1)

    for i in range(N):
        
        u_prime = u + sigma * torch.randn(nwalkers, ndim)
        X_prime, logdetJ_prime = flow.inverse(u_prime)
        J_prime = logdetJ_prime.sum(-1)
        Z_prime, Zl_prime, Zp_prime = logprob(torch_to_numpy(X_prime), return_torch=True)

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

    print(torch.mean(alpha), sigma)

    return X, Z, Zl, Zp, sigma


@torch.no_grad()
def LangevinMetropolis(logprob, flow, x, N, tau, target=0.57, adapt=True):

    def score(u):
        return - u

    def logq(up, u):
        return -0.5 / tau * torch.sum((up - u - tau/2.0 * score(u))**2, axis=1)

    nwalkers, ndim = x.shape
    
    X = torch.clone(x)
    Z, Zl, Zp = logprob(torch_to_numpy(x), return_torch=True)

    u, logdetJ = flow.forward(x)
    J = -logdetJ.sum(-1)

    for i in range(N):
        
        u_prime = u + tau / 2.0 * score(u) + tau**0.5 * torch.randn(nwalkers, ndim)
        X_prime, logdetJ_prime = flow.inverse(u_prime)
        J_prime = logdetJ_prime.sum(-1)
        Z_prime, Zl_prime, Zp_prime = logprob(torch_to_numpy(X_prime), return_torch=True)

        alpha = torch.minimum(torch.ones(len(X_prime)), torch.exp( Z_prime - Z + J_prime - J + logq(u, u_prime) - logq(u_prime, u) ))
        alpha[torch.isnan(alpha)] = 0.0
        mask = torch.rand(nwalkers) < alpha

        u[mask] = u_prime[mask]
        J[mask] = J_prime[mask]
        X[mask] = X_prime[mask]
        Z[mask] = Z_prime[mask]
        Zl[mask] = Zl_prime[mask]
        Zp[mask] = Zp_prime[mask]

        if adapt:
            tau_prime = tau + 1/(i+1) * (torch.mean(alpha) - target)
            if tau_prime > 1e-4:
                tau = tau_prime

    print(torch.mean(alpha), tau)

    return X, Z, Zl, Zp, tau