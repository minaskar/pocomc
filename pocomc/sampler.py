import numpy as np

from .mcmc import PreconditionedMetropolis, Metropolis
from .tools import resample_equal, _FunctionWrapper, torch_to_numpy, numpy_to_torch, get_ESS, ProgressBar
from .scaler import Reparameterise
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
                 corr_threshold=None,
                 target_accept=0.234,
                 flow_config=None,
                 train_config=None,
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

        # State variables
        self.u = None
        self.x = None
        self.J = None
        self.L = None
        self.P = None

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
        self.flow = Flow(self.ndim, flow_config, train_config)
        
        # Scaler
        if bounds is None:
            bounds = np.full((self.ndim, 2), np.nan)
        self.scaler = Reparameterise(bounds, scale, diagonal)
        self.rescale = rescale

        # temp
        self.ideal_scale = 2.38 / np.sqrt(ndim)
        self.scale = 2.38 / np.sqrt(ndim)
        self.threshold = threshold
        self.use_flow = False
        self.accept = target_accept
        self.target_accept = target_accept

        # temp 2
        self.corr_threshold = corr_threshold


    def run(self, x0, ess=0.95, nmin=5, nmax=1000, progress=True):

        # Run parameters
        self.ess = ess
        self.nmin = nmin
        self.nmax = nmax
        self.progress = progress

        # Set state parameters
        self.x = np.copy(x0)
        self.scaler.fit(self.x)
        self.u = self.scaler.forward(self.x)
        self.J = self.scaler.inverse(self.u)[1]
        self.P = self._logprior(self.x)
        self.L = self._loglike(self.x)
        self.saved_logl.append(self.L)
        self.ncall += len(self.x)

        assert np.allclose(self.scaler.inverse(self.u)[0], self.x)

        # Pre-train flow if required
        if self.threshold >= 1.0:
            self.use_flow = True
            self._train(self.u)

        # Save state
        self.saved_samples.append(self.x)
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
                                    accept=0.234,
                                    N=0,
                                    scale=1.0))
        
        # Run Sequential Monte Carlo
        while 1.0 - self.beta >= 1e-4:

            # Choose next beta based on CV of weights
            self._update_beta()

            # Resample x_prev, w
            self.u, self.x, self.J, self.L, self.P = self._resample(self.u,
                                                                    self.x,
                                                                    self.J,
                                                                    self.L,
                                                                    self.P)

            # Evolve particles using MCMC
            self.u, self.x, self.J, self.L, self.P = self._mutate(self.u,
                                                                  self.x,
                                                                  self.J,
                                                                  self.L,
                                                                  self.P)

            #if self.rescale:
            #    x = self.scaler.inverse(self.u)[0]
            #    self.scaler.fit(x)
            #    self.u = self.scaler.forward(x)

            # Train Precondiotoner
            self._train(self.u)

        self.saved_posterior_samples.append(self.x)
        
        self.pbar.close()

    
    def add_samples(self, N=1000, retrain=False, progress=True):
        self.progress = progress

        self.pbar = ProgressBar(self.progress)
        self.pbar.update_stats(dict(beta=self.beta,
                                    calls=self.ncall,
                                    ESS=self.ess,
                                    logZ=self.logz,
                                    accept=0,
                                    N=0,
                                    scale=0))

        iterations = int(np.ceil(N/len(self.u)))
        for _ in range(iterations):
            self.u = self._mutate(self.u)
            if retrain:
                self._train(self.u)
            self.saved_posterior_samples.append(self.scaler.inverse(self.u)[0])
            self.pbar.update_iter()

        self.pbar.close()
        #self.u = np.tile(self.u.T, multiply).T

    
    def _mutate(self, u, x, J, L, P):

        state_dict = dict(u=u,
                          x=x,
                          J=J,
                          L=L,
                          P=P,
                          beta=self.beta)

        function_dict = dict(loglike=self._loglike,
                             logprior=self._logprior,
                             scaler=self.scaler,
                             flow=self.flow)

        option_dict = dict(nmin=self.nmin,
                           nmax=self.nmax,
                           corr_threshold=self.corr_threshold,
                           sigma=self.scale,
                           progress_bar=self.pbar)

        if self.use_flow:
            results = PreconditionedMetropolis(state_dict,
                                               function_dict, 
                                               option_dict)
        else:
            results = Metropolis(state_dict,
                                 function_dict,
                                 option_dict)
            
        u = results.get('u')
        x = results.get('x')
        J = results.get('J')
        L = results.get('L')
        P = results.get('P')

        self.scale = results.get('scale')
        self.Nsteps = results.get('steps')
        self.accept = results.get('accept')
        
        self.ncall += self.Nsteps * len(x)

        self.saved_ncall.append(self.ncall)
        self.saved_accept.append(self.accept)
        self.saved_scale.append(self.scale/self.ideal_scale)
        self.saved_steps.append(self.Nsteps)

        return u, x, J, L, P


    def _train(self, x):
        if (self.scale < self.threshold * self.ideal_scale and self.t > 1) or self.use_flow:
            y = np.copy(x)
            np.random.shuffle(y)
            self.flow.fit(numpy_to_torch(y))
            self.use_flow = True
        else:
            pass


    def _resample(self, u, x, J, L, P):
        self.saved_samples.append(x)
        idx = resample_equal(np.arange(len(u)), np.exp(self.logw-np.max(self.logw))/np.sum(np.exp(self.logw-np.max(self.logw))))
        self.logw = 0.0

        return u[idx], x[idx], J[idx], L[idx], P[idx]

    
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
            self.logw = self.logw_prev + self.L * (beta - beta_prev)
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