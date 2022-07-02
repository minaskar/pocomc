import numpy as np
import torch

from .mcmc import PreconditionedMetropolis, Metropolis
from .tools import resample_equal, _FunctionWrapper, torch_to_numpy, numpy_to_torch, get_ESS, ProgressBar
from .scaler import Reparameterise
from .flow import Flow

class Sampler:
    r"""Main Precondioned Monte Carlo sampler class.

    Parameters
    ----------
    nparticles : int
        The total number of particles/walkers to use.
    ndim : int
        The total number of parameters/dimensions.
    loglikelihood : callable
        Function returning the log likelihood of a set
        of parameters.
    logprior : callable
        Function returning the log prior of a set
        of parameters.
    bounds : ``np.ndarray`` or None
        Array of shape ``(ndim, 2)`` holding the boundaries
        of parameters (default is ``bounds=None``). If a
        parameter is unbounded from below, above or both
        please provide ``None`` for the respective boundary.
    threshold : float
        The threshold value for the (normalised) proposal
        scale parameter below which normalising flow
        preconditioning (NFP) is enabled (default is
        ``threshold=1.0``, meaning that NFP is used all
        the time).
    scale : bool
        Scale
    rescale : bool
        Rescale
    diagonal : bool
        Diagonal
    loglikelihood_args : list
        Extra arguments to be passed into the loglikelihood
        (default is ``loglikelihood_args=None``).
    loglikelihood_kwargs : list
        Extra arguments to be passed into the loglikelihood
        (default is ``loglikelihood_kwargs=None``).
    logprior_args : list
        Extra arguments to be passed into the logprior
        (default is ``logprior_args=None``).
    logprior_kwargs : list
        Extra arguments to be passed into the logprior
        (default is ``logprior_kwargs=None``).
    vectorize_likelihood : bool
        Whether or not to vectorize the ``loglikelihood``
        calculation (default is ``vectorize_likelihood=False``).
    vectorize_logprior : bool
        Whether or not to vectorize the ``logprior``
        calculation (default is ``vectorize_prior=False``).
    pool : pool
        Provided ``MPI`` or ``multiprocessing`` pool for
        parallelisation (default is ``pool=None``).
    parallelize_prior : bool
        Whether or not to use the ``pool`` (if provided)
        for the ``logprior`` as well (default is 
        ``parallelize_prior=False``).
    flow_config : dict or ``None``
        Configuration of the normalizing flow (default is
        ``flow_config=None``).
    train_config : dict or ``None``
        Configuration for training the normalizing flow
        (default is ``train_config=None``).
    
    Attributes
    ----------
    results : dict
        Dictionary holding results. Includes the following
        properties: ``iter``, ``samples``, ``posterior_samples``,
        ``logw``, ``logl``, ``logz``, ``ess``, ``ncall``, 
        ``beta``, ``accept``, ``scale``, and ``steps``.
    """

    def __init__(self,
                 nparticles,
                 ndim,
                 loglikelihood,
                 logprior,
                 bounds=None,
                 periodic=None,
                 reflective=None,
                 threshold=1.0,
                 scale=True,
                 rescale=False,
                 diagonal=True,
                 loglikelihood_args=None,
                 loglikelihood_kwargs=None,
                 logprior_args=None,
                 logprior_kwargs=None,
                 vectorize_likelihood=False,
                 vectorize_prior=False,
                 pool=None,
                 parallelize_prior=False,
                 flow_config=None,
                 train_config=None,
                 random_state: int = None
                 ):
        if random_state is not None:
            np.random.seed(random_state)
            torch.manual_seed(random_state)

        self.nwalkers = nparticles
        self.ndim = ndim
        
        # Distributions
        self.loglikelihood = _FunctionWrapper(loglikelihood,
                                              loglikelihood_args, 
                                              loglikelihood_kwargs)
        self.logprior = _FunctionWrapper(logprior, 
                                         logprior_args, 
                                         logprior_kwargs)

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
        self.saved_posterior_logl = []
        self.saved_posterior_logp = []
        self.saved_logl = []
        self.saved_logp = []
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

        # Parallelism
        self.pool = pool
        if pool is None:
            self.distribute = map
        else:
            self.distribute = pool.map
        self.vectorize_likelihood = vectorize_likelihood
        self.vectorize_prior = vectorize_prior
        self.parallelize_prior = parallelize_prior

        # Flow
        self.flow = Flow(self.ndim, flow_config, train_config)
        self.threshold = threshold
        self.use_flow = False
        
        # Scaler
        self.scaler = Reparameterise(self.ndim, bounds, periodic, reflective, scale, diagonal)
        self.rescale = rescale

        # MCMC parameters
        self.ideal_scale = 2.38 / np.sqrt(ndim)
        self.scale = 2.38 / np.sqrt(ndim)
        self.accept = 0.234
        self.target_accept = 0.234

    def run(self,
            x0,
            ess = 0.95,
            gamma = 0.75,
            nmin = None,
            nmax = None,
            progress=True
            ):
        r"""Method that runs Preconditioned Monte Carlo.

        Parameters
        ----------
        x0 : ``np.ndarray``
            Array holding the initial positions of the particles. The initial
            positions must be sampled from the prior distribution.
        ess : float
            The effective sample size maintained during the run (default is
            `ess=0.95`).
        gamma : float
            Threshold for the correlation coefficient that is
            used to adaptively determine the number of MCMC
            steps (default is ``gamma=0.75``).
        nmin : int or None
            The minimum number of MCMC steps per iteration (default is ``nmin = ndim // 2``).
        nmax : int or None
            The maximum number of MCMC steps per iteration  (default is ``nmin = int(10 * ndim)``).
        progress : bool
            Whether or not to print progress bar (default is ``progress=True``).        
        """

        # Run parameters
        self.ess = ess
        self.gamma = gamma
        if nmin is None:
            self.nmin = self.ndim // 2
        else:
            self.nmin = int(nmin)
        if nmax is None:
            self.nmax = int(10 * self.ndim)
        else:
            self.nmax = int(nmax)
        self.progress = progress

        # Set state parameters
        self.x = np.copy(x0)
        self.scaler.fit(self.x)
        self.u = self.scaler.forward(self.x)
        self.J = self.scaler.inverse(self.u)[1]
        self.P = self._logprior(self.x)
        self.L = self._loglike(self.x)
        self.ncall += len(self.x)

        # Pre-train flow if required
        if self.threshold >= 1.0:
            self.use_flow = True
            self._train(self.u)

        # Save state
        self.saved_samples.append(self.x)
        self.saved_logl.append(self.L)
        self.saved_logp.append(self.P)
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

            # Rescale parameters
            if self.rescale:
                self.scaler.fit(self.x)
                self.u = self.scaler.forward(self.x)
                self.J = self.scaler.inverse(self.u)[1]

            # Train Precondiotoner
            self._train(self.u)

        self.saved_posterior_samples.append(self.x.copy())
        self.saved_posterior_logl.append(self.L.copy())
        self.saved_posterior_logp.append(self.P.copy())
        
        self.pbar.close()

    
    def add_samples(self,
                    N=1000,
                    retrain=False, 
                    progress=True
                    ):
        r"""Method that generates additional samples at the end of the run

        Parameters
        ----------
        N : int
            The number of additional samples (default is ``N=1000``).
        retrain : bool
            Whether or not to retrain the normalising flow preconditioner
            between iterations (default is ``retrain=False``).
        progress : bool
            Whether or not to print progress bar (default is ``progress=True``).
        """
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
            self.u, self.x, self.J, self.L, self.P = self._mutate(self.u,
                                                                  self.x,
                                                                  self.J,
                                                                  self.L,
                                                                  self.P)
            if retrain:
                self._train(self.u)
            self.saved_posterior_samples.append(self.x)
            self.saved_posterior_logl.append(self.L)
            self.saved_posterior_logp.append(self.P)
            self.pbar.update_iter()

        self.pbar.close()
        #self.u = np.tile(self.u.T, multiply).T

    
    def _mutate(self, u, x, J, L, P):

        state_dict = dict(u=u.copy(),
                          x=x.copy(),
                          J=J.copy(),
                          L=L.copy(),
                          P=P.copy(),
                          beta=self.beta)

        function_dict = dict(loglike=self._loglike,
                             logprior=self._logprior,
                             scaler=self.scaler,
                             flow=self.flow)

        option_dict = dict(nmin=self.nmin,
                           nmax=self.nmax,
                           corr_threshold=self.gamma,
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
            
        u = results.get('u').copy()
        x = results.get('x').copy()
        J = results.get('J').copy()
        L = results.get('L').copy()
        P = results.get('P').copy()

        self.scale = results.get('scale')
        self.Nsteps = results.get('steps')
        self.accept = results.get('accept')
        
        self.ncall += self.Nsteps * len(x)

        self.saved_ncall.append(self.ncall)
        self.saved_accept.append(self.accept)
        self.saved_scale.append(self.scale/self.ideal_scale)
        self.saved_steps.append(self.Nsteps)

        return u, x, J, L, P


    def _train(self, u):
        if (self.scale < self.threshold * self.ideal_scale and self.t > 1) or self.use_flow:
            y = np.copy(u)
            np.random.shuffle(y)
            self.flow.fit(numpy_to_torch(y))
            self.use_flow = True
        else:
            pass


    def _resample(self, u, x, J, L, P):
        self.saved_samples.append(x)
        self.saved_logl.append(L)
        self.saved_logp.append(P)
        w = np.exp(self.logw-np.max(self.logw))
        w /= np.sum(w)

        assert np.any(~np.isnan(self.logw))
        assert np.any(np.isfinite(self.logw))
        assert np.any(~np.isnan(w))
        assert np.any(np.isfinite(w))

        try:
            idx = resample_equal(np.arange(len(u)), w)
        except:
            idx = np.random.choice(np.arange(len(u)), p=w, size=len(w))
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
        if self.vectorize_likelihood:
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
            'samples' : np.vstack(self.saved_posterior_samples),
            'loglikelihood' : np.hstack(self.saved_posterior_logl),
            'logprior' : np.hstack(self.saved_posterior_logp),
            'logz' : np.array(self.saved_logz),
            'iter' : np.array(self.saved_iter),
            'x' : np.array(self.saved_samples),
            'logl' : np.array(self.saved_logl),
            'logp' : np.array(self.saved_logp),
            'logw' : np.array(self.saved_logw),
            'ess' : np.array(self.saved_ess),
            'ncall' : np.array(self.saved_ncall),
            'beta' : np.array(self.saved_beta),
            'accept' : np.array(self.saved_accept),
            'scale' : np.array(self.saved_scale),
            'steps' : np.array(self.saved_steps)
        }

        return results

    def bridge_sampling(self, tolerance=1e-10, maxiter=1000, thin=1):

        x = self.results.get("samples")[::thin]
        l = self.results.get("loglikelihood")[::thin]
        p = self.results.get("logprior")[::thin]
        
        N1 = len(x)
        N2 = len(x)

        s1 = N1 / (N1+N2)
        s2 = N2 / (N1+N2)

        import torch

        u_prop, logg_i = self.flow.sample(size=N2)
        x_prop, J_prop = self.scaler.inverse(torch_to_numpy(u_prop))
        u_prop = torch_to_numpy(u_prop)
        logg_i = torch_to_numpy(logg_i) + J_prop

        u = self.scaler.forward(x)
        x, J = self.scaler.inverse(u)
        logg_j = torch_to_numpy(self.flow.logprob(numpy_to_torch(u))) - J

        logp_i = self._loglike(x_prop) + self._logprior(x_prop) - J_prop
        logp_j = p + l

        logl1j = logp_j - logg_j
        logl2i = logp_i - logg_i

        lstar = max(np.max(logl1j), np.max(logl2i))

        l1j = np.exp(logl1j - lstar)
        l2i = np.exp(logl2i - lstar)

        r = 1.0
        r0 = 0.0
        cnt = 1
        while np.abs(r-r0) > tolerance or cnt <= maxiter:
            r0 = r
            A = np.mean(l2i / (s1 * l2i + s2 * r0))
            B = np.mean(1.0 / (s1 * l1j + s2 * r0))
            r = A / B
            cnt += 1

        return np.log(r) + lstar