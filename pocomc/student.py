import numpy
from scipy import optimize
from scipy import special

def fit_mvstud(data, tolerance=1e-6, max_iter=100):
    def opt_nu(delta_iobs, nu):
        def func0(nu):
            w_iobs = (nu + dim) / (nu + delta_iobs)
            f = -special.psi(nu/2) + numpy.log(nu/2) + numpy.sum(numpy.log(w_iobs))/n - numpy.sum(w_iobs)/n + 1 + special.psi((nu+dim)/2) - numpy.log((nu+dim)/2)
            return f

        if func0(1e6) >= 0:
            nu = numpy.inf
        else:
            nu = optimize.brentq(func0, 1e-6, 1e6)
        return nu

    data = data.T
    (dim,n) = data.shape
    mu = numpy.array([numpy.median(data,1)]).T
    Sigma = numpy.cov(data)*(n-1)/n + 1e-1*numpy.eye(dim)
    nu = 20

    last_nu = 0
    i = 0
    while numpy.abs(last_nu - nu) > tolerance and i < max_iter:
        i += 1
        diffs = data - mu
        delta_iobs = numpy.sum(diffs * numpy.linalg.solve(Sigma,diffs), 0)
        
        # update nu
        last_nu = nu
        nu = opt_nu(delta_iobs, nu)
        if nu == numpy.inf:
            return mu.T[0], Sigma, nu

        w_iobs = (nu + dim) / (nu + delta_iobs)

        # update Sigma
        Sigma = numpy.dot(w_iobs*diffs, diffs.T) / n

        # update mu
        mu = numpy.sum(w_iobs * data, 1) / sum(w_iobs)
        mu = numpy.array([mu]).T

    return mu.T[0], Sigma, nu
