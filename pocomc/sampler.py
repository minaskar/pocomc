import numpy as np
from .mcmc import RandomWalkMetropolis, LangevinMetropolis
from .tools import get_ESS_CV, resample_equal, progress_bar, _FunctionWrapper, torch_to_numpy, numpy_to_torch
from .scaler import IdentityScaler, StandardScaler
from .flow import Flow

class Sampler:

    def __init__(self,
                 loglikelihood,
                 logprior,
                 args=None,
                 kwargs=None,
                 resample=True,
                 vectorize=False,
                 vectorize_prior=False,
                 pool=None,
                 parallelize_prior=False,
                 scale_input=True,
                 rescale_input=True,
                 ):

        self.tau = 0.5
        
        # Distributions
        self.loglikelihood = _FunctionWrapper(loglikelihood, args, kwargs)
        self.logprior = logprior

        # Sampling
        self.ncall = 0
        self.t = 0
        self.beta = 0.0
        self.logw = 0.0
        self.sum_logw = 0.0
        self.logz = 0.0
        self.sigma_logz = 0.0
        self.resample = resample

        # Results
        self.saved_iter = []
        self.saved_samples = []
        self.saved_logl = []
        self.saved_logw = []
        self.saved_logz = []
        self.saved_slogz = []
        self.saved_cv = []
        self.saved_ess = []
        self.saved_ncall = []
        self.saved_beta = []

        # parallelism
        self.pool = pool
        self.vectorize = vectorize
        self.vectorize_prior = vectorize_prior
        self.parallelize_prior = parallelize_prior

        # Scaler
        if scale_input:
            self.scaler = StandardScaler()
        else:
            self.scaler = IdentityScaler()
        self.rescale_input = rescale_input


    def run(self, x0, cv=1.0, niter=5, progress=True):

        self.niter = niter
        self.cv = cv 

        x = np.copy(x0)
        x = self.scaler.fit_transform(x)
        self.nwalkers, self.ndim = np.shape(x)

        # Initial training of the flow
        self.flow = Flow(self.ndim)
        history = self.flow.fit(x)

        self.ess = self.nwalkers
        self.saved_samples.append(x)
        self.saved_iter.append(self.t)
        self.saved_beta.append(self.beta)
        self.saved_logw.append(np.zeros(self.nwalkers))
        self.saved_logz.append(self.logz)
        self.saved_slogz.append(0.0)
        self.saved_ess.append(self.nwalkers)

        # Initialise progress bar
        pbar = progress_bar(progress)
        
        self.loglike = self._loglike(x)
        self.saved_logl.append(self.loglike)
        self.ncall += len(x)
        
        while 1.0 - self.beta >= 1e-4:
    
            # Update iteration index
            self.t += 1
            self.saved_iter.append(self.t)

            # Choose next beta based on CV of weights
            self.beta = self._choose_beta()

            # Compute Model Evidence
            self.logz += np.mean(self.logw)
            self.sigma_logz = np.std(self.sum_logw) / np.sqrt(len(self.sum_logw) - 1.0)
            self.saved_logz.append(self.logz)
            self.saved_slogz.append(self.sigma_logz)

            # Resample x_prev, w
            if self.resample:
                x = self._resample(x)

            if self.rescale_input:
                x = self.scaler.refit_transform(x)

            # Evolve particles using MCMC
            x = self._mutate(x)

            # Print progress bar
            pbar.update(self.beta, self.ncall, self.ess_est, self.cv_est, self.logz, [np.round(self.sigma_logz,4), np.round(np.abs(self.sigma_logz / self.logz),4)])
            
        pbar.close()

        return self.scaler.inverse_transform(x), self.logz


    def _mutate(self, x_prev):
        
        x, Z, Zl, Zp, self.tau = RandomWalkMetropolis(self._logprob, self.flow, numpy_to_torch(x_prev), self.niter, self.tau)
        #x, Z, Zl, Zp, self.tau = LangevinMetropolis(self._logprob, self.flow, numpy_to_torch(x_prev), self.niter, self.tau)
        
        self.loglike = np.copy(torch_to_numpy(Zl))
        self.ncall += self.niter * len(x)

        self.flow.fit(x)

        return torch_to_numpy(x)


    def _resample(self, x_prev):
        self.saved_samples.append(x_prev)
        x_prev = resample_equal(x_prev, np.exp(self.logw-np.max(self.logw))/np.sum(np.exp(self.logw-np.max(self.logw))))
        self.logw = 0.0

        return x_prev

    
    def _choose_beta(self):
        beta_prev = np.copy(self.beta)
        beta_max = 1.0
        beta_min = np.copy(beta_prev)
        self.logw_prev = np.copy(self.logw)
        while True:

            beta = (beta_max + beta_min) * 0.5
            self.logw = self.logw_prev + self.loglike * (beta - beta_prev)
            self.ess_est, self.cv_est = get_ESS_CV(self.logw)

            # Test
            logz = self.logz + np.mean(self.logw)
            sum_logw = self.sum_logw + self.logw
            sigma_logz = np.std(sum_logw) / np.sqrt(len(sum_logw) - 1.0)
            # ---

            if len(self.saved_beta) > 1:
                dbeta = self.saved_beta[-1] - self.saved_beta[-2]

                if 1.0 - beta < dbeta * 0.1:
                    beta = 1.0

            if (np.abs(self.cv_est-self.cv) < 0.001 * self.cv or beta == 1.0):
                self.saved_beta.append(beta)
                self.saved_logw.append(self.logw)
                self.sum_logw += self.logw
                self.saved_cv.append(self.cv_est)
                self.saved_ess.append(self.ess_est)
                return beta
            elif self.cv_est > self.cv:
                beta_max = beta 
            else:
                beta_min = beta
        

    def _logprior(self, x):
        x = self.scaler.inverse_transform(x)
        if self.vectorize_prior:
            return self.logprior(x)
        elif self.parallelize_prior and self.pool is not None:
            return np.array(list(self.pool.map(self.logprior, x)))
        else:
            return np.array(list(map(self.logprior, x)))


    def _loglike(self, x):
        x = self.scaler.inverse_transform(x)
        if self.vectorize:
            return self.loglikelihood(x)
        elif self.pool is not None:
            return np.array(list(self.pool.map(self.loglikelihood, x)))
        else:
            return np.array(list(map(self.loglikelihood, x)))


    def _logprob(self, x, return_torch=False):
        Zp = self._logprior(x)
        Zl = self._loglike(x)

        Zl[np.isnan(Zp)] = -np.inf
        Zl[~np.isfinite(Zp)] = -np.inf

        Z = Zp + self.beta * Zl
        if return_torch:
            return numpy_to_torch(Z), numpy_to_torch(Zl), numpy_to_torch(Zp)
        else:
            return Z, Zl, Zp
        

    def __getstate__(self):
        """Get state information for pickling."""

        state = self.__dict__.copy()

        try:
            #remove random module
            #del state['rstate']

            # deal with pool
            if state['pool'] is not None:
                del state['pool']  # remove pool
                del state['distribute']  # remove `pool.map` function hook
        except:
            pass

        return state

    @property
    def results(self):
        
        results = {
            'iter' : np.array(self.saved_iter),
            'samples' : np.array(self.saved_samples),
            'logl' : np.array(self.saved_samples),
            'logw' : np.array(self.saved_logw),
            'logz' : np.array(self.saved_logz),
            'cv' : np.array(self.saved_cv),
            'ess' : np.array(self.saved_ess),
            'ncall' : np.array(self.saved_ncall),
            'beta' : np.array(self.saved_beta)
        }

        return results