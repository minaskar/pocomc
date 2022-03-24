import numpy as np

from .mcmc import PreconditionedMetropolis, Metropolis
from .tools import resample_equal, _FunctionWrapper, torch_to_numpy, numpy_to_torch, get_ESS, ProgressBar
from .scaler import Reparameterise

from pocomc.sinf.base import SINFInterface
from .flow import Flow

class Sampler:

    def __init__(self,
                 nwalkers,
                 ndim,
                 loglikelihood,
                 logprior,
                 bounds=None,
                 threshold=0.75,
                 scale=True,
                 rescale=False,
                 diagonal=True,
                 loglikelihood_args=None,
                 loglikelihood_kwargs=None,
                 logprior_args=None,
                 logprior_kwargs=None,
                 vectorize=False,
                 vectorize_prior=False,
                 pool=None,
                 parallelize_prior=False,
                 # Temp
                 alpha=(0,0.97),
                 corr_threshold=0.01,
                 target_accept=0.234,
                 # MAF
                 use_maf=False,
                 flow_config=None,
                 train_config=None,
                 lazy=False,
                 ):

        self.nwalkers = nwalkers
        self.ndim = ndim
        
        # Distributions
        self.loglikelihood = _FunctionWrapper(loglikelihood, loglikelihood_args, loglikelihood_kwargs)
        self.logprior = _FunctionWrapper(logprior, logprior_args, logprior_kwargs)

        # Sampling
        self.ncall = 0
        self.t = 0
        self.beta = 0.0
        self.logw = 0.0
        self.sum_logw = 0.0
        self.logz = 0.0

        # Results
        self.saved_iter = []
        self.saved_samples = []
        self.saved_posterior_samples = []
        self.saved_logl = []
        self.saved_logw = []
        self.saved_logz = []
        self.saved_ess = []
        self.saved_ncall = []
        self.saved_beta = []
        self.saved_accept = []
        self.saved_scale = []
        self.saved_steps = []

        # parallelism
        self.pool = pool
        if pool is None:
            self.distribute = map
        else:
            self.distribute = pool.map
        self.vectorize = vectorize
        self.vectorize_prior = vectorize_prior
        self.parallelize_prior = parallelize_prior

        # Flow
        self.use_maf = use_maf
        if self.use_maf:
            self.flow = Flow(self.ndim, flow_config, train_config)
        else:
            self.flow = SINFInterface()

        # Scaler
        if bounds is None:
            bounds = np.full((self.ndim, 2), np.nan)
        self.scaler = Reparameterise(bounds, scale, diagonal)
        self.rescale = rescale

        # temp
        self.ideal_scale = 2.38/np.sqrt(ndim)
        self.scale = 2.38/np.sqrt(ndim)
        self.threshold = threshold
        self.use_flow = False
        self.accept = target_accept
        self.target_accept = target_accept

        # temp 2
        self.alpha = alpha
        self.corr_threshold = corr_threshold


    def run(self, x0, ess=0.95, niter=4, nmin=1, nmax=500, progress=True):

        self.ess = ess
        self.niter = niter
        self.nmin = nmin
        self.nmax = nmax
        self.progress = progress

        x = np.copy(x0)

        self.loglike = self._loglike(x)
        self.saved_logl.append(self.loglike)
        self.ncall += len(x)

        self.scaler.fit(x)
        self.u = self.scaler.forward(x)

        self.saved_samples.append(x)
        self.saved_iter.append(self.t)
        self.saved_beta.append(self.beta)
        self.saved_logw.append(np.zeros(self.nwalkers))
        self.saved_logz.append(self.logz)
        self.saved_ess.append(self.ess)
        self.saved_ncall.append(self.ncall)
        self.saved_accept.append(self.accept)
        self.saved_scale.append(self.scale/self.ideal_scale)
        self.saved_steps.append(0)

        # Initialise progress bar
        self.pbar = ProgressBar(self.progress)
        self.pbar.update_stats(dict(beta=self.beta,
                               calls=self.ncall,
                               ESS=self.ess,
                               logZ=self.logz,
                               accept=0,
                               N=0,
                               scale=0,
                              )
                         )
        
        while 1.0 - self.beta >= 1e-4:

            # Choose next beta based on CV of weights
            self._update_beta()

            # Resample x_prev, w
            self.u = self._resample(self.u)

            # Evolve particles using MCMC
            self.u = self._mutate(self.u)

            if self.rescale:
                x = self.scaler.inverse(self.u)[0]
                self.scaler.fit(x)
                self.u = self.scaler.forward(x)

            # Train Precondiotoner
            self._train(self.u)

        self.saved_posterior_samples.append(self.scaler.inverse(self.u)[0])
        
        self.pbar.close()

    
    def add_samples(self, N=1000, retrain=False):

        self.pbar = ProgressBar(self.progress)
        self.pbar.update_stats(dict(beta=self.beta,
                               calls=self.ncall,
                               ESS=self.ess,
                               logZ=self.logz,
                               accept=0,
                               N=0,
                               scale=0,
                              ))

        iterations = int(np.ceil(N/len(self.u)))
        for _ in range(iterations):
            self.u = self._mutate(self.u)
            if retrain:
                self._train(self.u)
            self.saved_posterior_samples.append(self.scaler.inverse(self.u)[0])
            self.pbar.update_iter()

        self.pbar.close()
        #self.u = np.tile(self.u.T, multiply).T

    
    def _mutate(self, x_prev):

        if self.use_flow:
            results = PreconditionedMetropolis(self._logprob,
                                               self.flow,
                                               x_prev,
                                               self.nmin,
                                               self.nmax,
                                               self.scale,
                                               self.target_accept,
                                               True,
                                               self.corr_threshold,
                                               self.pbar,
                                               self.use_maf)
        else:
            results = Metropolis(self._logprob,
                                 x_prev,
                                 self.nmin,
                                 self.nmax,
                                 self.scale,
                                 np.cov(x_prev.T),
                                 self.target_accept,
                                 True,
                                 self.corr_threshold,
                                 self.pbar)
            
        x = results.get('X')
        Zl = results.get('Zl')
        self.scale = results.get('scale')
        self.Nsteps = results.get('steps')
        self.accept = results.get('accept')
        
        
        self.loglike = np.copy(Zl)
        self.ncall += self.Nsteps * len(x)

        self.saved_ncall.append(self.ncall)
        self.saved_accept.append(self.accept)
        self.saved_scale.append(self.scale/self.ideal_scale)
        self.saved_steps.append(self.Nsteps)


        return x


    def _train(self, x):
        if (self.scale < self.threshold * self.ideal_scale and self.t > 1) or self.use_flow:
            if self.use_flow:
                if self.use_maf:
                    self.flow.fit(numpy_to_torch(x))
                else:
                    self.flow.train_flow(numpy_to_torch(x), val_frac=0.2, alpha=self.alpha, Whiten=True)
            else:
                if self.use_maf:
                    self.flow.fit(numpy_to_torch(x))
                else:
                    self.flow.create_flow(numpy_to_torch(x), val_frac=0.2, alpha=self.alpha, Whiten=True)
                self.use_flow = True
        else:
            pass


    def _resample(self, x_prev):
        self.saved_samples.append(self.scaler.inverse(x_prev)[0])
        x_prev = resample_equal(x_prev, np.exp(self.logw-np.max(self.logw))/np.sum(np.exp(self.logw-np.max(self.logw))))
        self.logw = 0.0

        return x_prev

    
    def _update_beta(self):

        # Update iteration index
        self.t += 1
        self.saved_iter.append(self.t)
        self.pbar.update_iter()

        beta_prev = np.copy(self.beta)
        beta_max = 1.0
        beta_min = np.copy(beta_prev)
        self.logw_prev = np.copy(self.logw)

        while True:

            beta = (beta_max + beta_min) * 0.5
            self.logw = self.logw_prev + self.loglike * (beta - beta_prev)
            self.ess_est = get_ESS(self.logw)

            if len(self.saved_beta) > 1:
                dbeta = self.saved_beta[-1] - self.saved_beta[-2]

                if 1.0 - beta < dbeta * 0.1:
                    beta = 1.0

            if (np.abs(self.ess_est-self.ess) < min(0.001 * self.ess, 0.001) or beta == 1.0):
                self.saved_beta.append(beta)
                self.saved_logw.append(self.logw)
                self.sum_logw += self.logw
                self.saved_ess.append(self.ess_est)
                self.beta = beta
                self.pbar.update_stats(dict(beta=self.beta, ESS=self.ess_est))
                # Update evidence 
                self.logz += np.mean(self.logw)
                self.saved_logz.append(self.logz)
                self.pbar.update_stats(dict(logZ=self.logz))
                break

            elif self.ess_est < self.ess:
                beta_max = beta 
            else:
                beta_min = beta
        

    def _logprior(self, x):
        if self.vectorize_prior:
            return self.logprior(x)
        elif self.parallelize_prior and self.pool is not None:
            return np.array(list(self.distribute(self.logprior, x)))
        else:
            return np.array(list(map(self.logprior, x)))


    def _loglike(self, x):
        if self.vectorize:
            return self.loglikelihood(x)
        elif self.pool is not None:
            return np.array(list(self.distribute(self.loglikelihood, x)))
        else:
            return np.array(list(map(self.loglikelihood, x)))


    def _logprob(self, u, return_torch=False):
        x, J = self.scaler.inverse(u)

        Zp = self._logprior(x)
        Zl = self._loglike(x)

        Zl[np.isnan(Zl)] = -np.inf
        Zl[np.isnan(Zp)] = -np.inf
        Zl[~np.isfinite(Zp)] = -np.inf

        Z = Zp + self.beta * Zl
        if return_torch:
            return numpy_to_torch(Z), numpy_to_torch(Zl), numpy_to_torch(Zp), numpy_to_torch(J)
        else:
            return Z, Zl, Zp, J
        

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
            'posterior_samples' : np.vstack(self.saved_posterior_samples),
            'logl' : np.array(self.saved_samples),
            'logw' : np.array(self.saved_logw),
            'logz' : np.array(self.saved_logz),
            'ess' : np.array(self.saved_ess),
            'ncall' : np.array(self.saved_ncall),
            'beta' : np.array(self.saved_beta),
            'accept' : np.array(self.saved_accept),
            'scale' : np.array(self.saved_scale),
            'steps' : np.array(self.saved_steps)
        }

        return results