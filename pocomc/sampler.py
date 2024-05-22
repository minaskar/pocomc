from pathlib import Path
from typing import Union

import dill
import numpy as np
import torch

from .mcmc import preconditioned_pcn, preconditioned_rwm, pcn, rwm
from .tools import systematic_resample, FunctionWrapper, numpy_to_torch, torch_to_numpy, trim_weights, ProgressBar, flow_numpy_wrapper
from .scaler import Reparameterize
from .flow import Flow
from .particles import Particles
from .geometry import Geometry
from .threading import configure_threads

class Sampler:
    r"""Preconditioned Monte Carlo class.

    Parameters
    ----------
    prior : callable
        Class implementing the prior distribution.
    likelihood : callable
        Function returning the log likelihood of a set of parameters.
    n_dim : int
        The total number of parameters/dimensions (Optional as it can be infered from the prior class).
    n_ess : int
        The effective sample size maintained during the run (default is ``n_ess=512``). Higher values
        lead to more accurate results but also increase the computational cost. 
    n_active : int
        The number of active particles (default is ``n_active=256``). It must be smaller than ``n_ess``.
        This is the number of particles that are evolved using MCMC at each iteration. If a pool is provided,
        the number of active particles should be a multiple of the number of processes in the pool to ensure
        efficient parallelisation.
    likelihood_args : list
        Extra arguments to be passed to likelihood (default is ``likelihood_args=None``). Example:
        ``likelihood_args=[data]``.
    likelihood_kwargs : dict
        Extra arguments to be passed to likelihood (default is ``likelihood_kwargs=None``). Example:
        ``likelihood_kwargs={"data": data}``.
    vectorize : bool
        If True, vectorize ``likelihood`` calculation (default is ``vectorize=False``). If False,
        the likelihood is calculated for each particle individually. If ``vectorize=True``, the likelihood
        is calculated for all particles simultaneously. This can lead to a significant speed-up if the likelihood
        function is computationally expensive. However, it requires that the likelihood function can handle
        arrays of shape ``(n_active, n_dim)`` as input and return an array of shape ``(n_active,)`` as output.
    pool : pool
        Provided ``MPI`` or ``multiprocessing`` pool for parallelisation (default is ``pool=None``).
        For ``MPI``, the pool should be an instance of ``mpi4py.futures.MPIPoolExecutor``. For
        ``multiprocessing``, the pool should be an instance of ``multiprocessing.Pool``. If a pool is provided,
        the number of active particles should be a multiple of the number of processes in the pool to ensure
        efficient parallelisation. If ``pool=None``, the code runs in serial mode. When a pool is provided,
        please ensure that the likelihood function is picklable. 
    pytorch_threads : int
        Maximum number of threads to use for torch. If ``None`` torch uses all
        available threads while training the normalizing flow (default is ``pytorch_threads=1``). 
    flow : ``torch.nn.Module`` or ``None``
        Normalizing flow (default is ``None``). The default is a Masked Autoregressive Flow
        (MAF) with 6 blocks of 3x(3xn_dim) layers and residual connections.
    train_config : dict or ``None``
        Configuration for training the normalizing flow
        (default is ``train_config=None``). Options include a dictionary with the following
        keys: ``"validation_split"``, ``"epochs"``, ``"batch_size"``, ``"patience"``,
        ``"learning_rate"``, ``"annealing"``, ``"gaussian_scale"``, ``"laplace_scale"``,
        ``"noise"``, ``"shuffle"``, ``"clip_grad_norm"``.
    precondition : bool
        If True, use preconditioned MCMC (default is ``precondition=True``). If False,
        use standard MCMC without normalizing flow. The use of preconditioned MCMC is
        recommended as it is more efficient and scales better with the number of parameters. 
        However, it requires the use of a normalizing flow and the training of the flow
        can be computationally expensive. If ``precondition=False``, the normalizing flow
        is not used and the sampler runs in standard mode. This works well for targets that
        are not multimodal or have strong non-linear correlations between parameters.
    n_prior : int
        Number of prior samples to draw (default is ``n_prior=2*(n_ess//n_active)*n_active``).
    sample : ``str``
        Type of MCMC sampler to use (default is ``sample="pcn"``). Options are
        ``"pcn"`` (Preconditioned Crank-Nicolson) or ``"rwm"`` (Random-Walk Metropolis).
        Preconditioned Crank-Nicolson is the default and recommended sampler for PMC as it
        is more efficient and scales better with the number of parameters.
    n_steps : int
        Number of MCMC steps after logP plateau (default is ``n_steps=n_dim//2``). This is used
        for early stopping of MCMC. Higher values can lead to better exploration but also
        increase the computational cost. If ``n_steps=None``, the default value is ``n_steps=n_dim//2``.
    n_max_steps : int
        Maximum number of MCMC steps (default is ``n_max_steps=10*n_dim``).
    resample : ``str``
        Resampling scheme to use (default is ``resample="systematic"``). Options are
        ``"systematic"`` (systematic resampling) or ``"multinomial"`` (multinomial resampling).
    output_dir : ``str`` or ``None``
        Output directory for storing the state files of the
        sampler. Default is ``None`` which creates a ``states``
        directory. Output files can be used to resume a run.
    output_label : ``str`` or ``None``
        Label used in state files. Defaullt is ``None`` which
        corresponds to ``"pmc"``. The saved states are named
        as ``"{output_dir}/{output_label}_{i}.state"`` where
        ``i`` is the iteration index.  Output files can be
        used to resume a run.
    random_state : int or ``None``
        Initial random seed.
    """

    def __init__(self,
                 prior: callable,
                 likelihood: callable,
                 n_dim: int = None,
                 n_ess: int = 512,
                 n_active: int = 256,
                 likelihood_args: list = None,
                 likelihood_kwargs: dict = None,
                 vectorize: bool = False,
                 pool=None,
                 pytorch_threads=1,
                 flow=None,
                 train_config: dict = None,
                 precondition: bool = True,
                 dynamic: bool = False,
                 n_prior: int = None,
                 sample: str = None,
                 n_steps: int = None,
                 n_max_steps: int = None,
                 resample: str = None,
                 output_dir: str = None,
                 output_label: str = None,
                 random_state: int = None):

        # Random seed
        if random_state is not None:
            np.random.seed(random_state)
            torch.manual_seed(random_state)
        self.random_state = random_state

        # Configure threads
        configure_threads(pytorch_threads=pytorch_threads)

        # Prior
        self.prior = prior
        self.log_prior = self.prior.logpdf
        self.sample_prior = self.prior.rvs
        self.bounds = self.prior.bounds

        # Log likelihood
        self.log_likelihood = FunctionWrapper(
            likelihood,
            likelihood_args,
            likelihood_kwargs
        )

        # Number of parameters
        if n_dim is None:
            self.n_dim = self.prior.dim
        else:
            self.n_dim = int(n_dim)

        # Effective Sample Size
        self.n_ess = int(n_ess)

        # Number of active particles
        self.n_active = int(n_active)

        # Number of MCMC steps after logP plateau
        if n_steps is None:
            self.n_steps = int(self.n_dim//2)
        else:
            self.n_steps = int(n_steps)

        # Maximum number of MCMC steps
        if n_max_steps is None:
            self.n_max_steps = 10 * self.n_steps
        else:
            self.n_max_steps = int(n_max_steps)

        # Total ESS for termination
        self.n_total = None

        # Particle manager
        self.particles = Particles(n_active, n_dim)

        # Sampling
        self.t = 0

        # Parallelism
        self.pool = pool
        if pool is None:
            self.distribute = map
        else:
            self.distribute = pool.map
        self.vectorize_likelihood = vectorize

        # Geometry
        self.u_geometry = Geometry()
        self.theta_geometry = Geometry()

        # Flow
        self.flow = Flow(self.n_dim, flow)
        self.train_config = dict(validation_split=0.5,
                                 epochs=2000,
                                 batch_size=np.minimum(int(n_ess)//2, 512),
                                 patience=50,
                                 learning_rate=1e-3,
                                 annealing=False,
                                 gaussian_scale=None,
                                 laplace_scale=None,
                                 noise=None,
                                 shuffle=True,
                                 clip_grad_norm=1.0,
                                 verbose=0,
                                )
        if train_config is not None:
            for key in train_config.keys():
                self.train_config[key] = train_config[key]

        # Scaler
        self.scaler = Reparameterize(self.n_dim, bounds=self.bounds)

        # Output
        if output_dir is None:
            self.output_dir = Path("states")
        else:
            self.output_dir = output_dir
        if output_label is None:
            self.output_label = "pmc"
        else:
            self.output_label = output_label

        # Other
        self.preconditioned = precondition

        self.dynamic = dynamic

        if sample is None:
            self.sample = 'pcn'
        elif sample in ['pcn']:
            self.sample = 'pcn'
        elif sample in ['rwm', 'mh']:
            self.sample = 'rwm'

        if resample is None:
            self.resample = 'systematic'
        elif resample in ['systematic', 'systematic_resample', 'systematic_resampling', 'syst']:
            self.resample = 'systematic'
        elif resample in ['multinomial', 'multinomial_resample', 'multinomial_resampling', 'mult']:
            self.resample = 'multinomial'

        # Prior samples to draw
        if n_prior is None:
            self.n_prior = int(2 * np.maximum(self.n_ess//self.n_active, 1) * self.n_active)
        else:
            self.n_prior = int(np.maximum(n_prior/self.n_active, 1) * self.n_active)
        self.prior_samples = None

        self.logz = None
        self.logz_err = None

        self.current_particles = None
        self.warmup = True
        self.calls = 0
        
        self.progress = None
        self.pbar = None

    def run(self,
            n_total: int = 5000,
            n_evidence: int = 5000,
            progress: bool = True,
            resume_state_path: Union[str, Path] = None,
            save_every: int = None):
        r"""Run Preconditioned Monte Carlo.

        Parameters
        ----------
        n_total : int
            The total number of effectively independent samples to be
            collected (default is ``n_total=5000``).
        n_evidence : int
            The number of importance samples used to estimate the
            evidence (default is ``n_evidence=5000``). If ``n_evidence=0``,
            the evidence is not estimated using importance sampling and the
            SMC estimate is used instead. If ``preconditioned=False``, 
            the evidence is estimated using SMC and ``n_evidence`` is ignored.
        progress : bool
            If True, print progress bar (default is ``progress=True``).
        resume_state_path : ``Union[str, Path]``
            Path of state file used to resume a run. Default is ``None`` in which case
            the sampler does not load any previously saved states. An example of using
            this option to resume or continue a run is e.g. ``resume_state_path = "states/pmc_1.state"``.
        save_every : ``int`` or ``None``
            Argument which determines how often (i.e. every how many iterations) ``pocoMC`` saves
            state files to the ``output_dir`` directory. Default is ``None`` in which case no state
            files are stored during the run.
        """
        if resume_state_path is not None:
            self.load_state(resume_state_path)
            t0 = self.t
            # Initialise progress bar
            self.pbar = ProgressBar(self.progress)
            self.pbar.update_stats(dict(calls=self.particles.get("calls", -1),
                                        beta=self.particles.get("beta", -1),
                                        logZ=self.particles.get("logz", -1)))
        else:
            t0 = self.t
            # Run parameters
            self.n_total = int(n_total)
            self.progress = progress

            # Initialise progress bar
            self.pbar = ProgressBar(self.progress)

        # Initialise particles
        if self.prior_samples is None:
            self.prior_samples = self.sample_prior(self.n_prior)
            self.scaler.fit(self.prior_samples)

        if self.warmup:
            for i in range(self.n_prior//self.n_active):
                if save_every is not None:
                    if (self.t - t0) % int(save_every) == 0 and self.t != t0:
                        self.save_state(Path(self.output_dir) / f'{self.output_label}_{self.t}.state')
                # Set state parameters
                x = self.prior_samples[i*self.n_active:(i+1)*self.n_active]
                u = self.scaler.forward(x)
                logdetj = self.scaler.inverse(u)[1]
                logp = self.log_prior(x)
                logl = self._log_like(x)
                self.calls += self.n_active

                self.current_particles = dict(u=u,x=x,logl=logl,logp=logp,logdetj=logdetj,
                                    logw=-1e300 * np.ones(self.n_active), iter=self.t,
                                    calls=self.calls, steps=1, efficiency=1.0, ess=0.0, 
                                    accept=1.0, beta=0.0, logz=0.0)
                
                self.particles.update(self.current_particles)

                self.pbar.update_stats(dict(calls=self.particles.get("calls", -1), 
                                            beta=self.particles.get("beta", -1), 
                                            logZ=self.particles.get("logz", -1)))
                
                self.pbar.update_iter()

                self.t += 1
            self.warmup = False

        # Run Sequential Monte Carlo
        while self._not_termination(self.current_particles):
            if save_every is not None:
                if (self.t - t0) % int(save_every) == 0 and self.t != t0:
                    self.save_state(Path(self.output_dir) / f'{self.output_label}_{self.t}.state')

            # Choose next beta based on ESS of weights
            self.current_particles = self._reweight(self.current_particles)

            # Train Preconditioner
            self.current_particles = self._train(self.current_particles)

            # Resample particles
            self.current_particles = self._resample(self.current_particles)

            # Evolve particles using MCMC
            self.current_particles = self._mutate(self.current_particles)   

            # Save particles
            self.particles.update(self.current_particles)

        # Compute evidence
        if n_evidence > 0 and self.preconditioned:
            self._compute_evidence(int(n_evidence))
        else:
            _, self.logz = self.particles.compute_logw_and_logz(1.0)
            self.logz_err = None

        self.pbar.close()

    def _not_termination(self, current_particles):
        """
        Check if termination criterion is satisfied.

        Parameters
        ----------
        current_particles : dict
            Dictionary containing the current particles.
        
        Returns
        -------
        termination : bool
            True if termination criterion is not satisfied.
        """
        log_weights, _ = self.particles.compute_logw_and_logz(1.0)
        weights = np.exp(log_weights - np.max(log_weights))
        weights /= np.sum(weights)
        ess = 1.0 / np.sum(weights**2.0)

        return 1.0 - current_particles.get("beta") >= 1e-4 or ess < self.n_total

    
    def _mutate(self, current_particles):
        """
        Evolve particles using MCMC.

        Parameters
        ----------
        current_particles : dict
            Dictionary containing the current particles.
        
        Returns
        -------
        current_particles : dict
            Dictionary containing the updated particles.
        """
        state_dict = dict(
            u=current_particles.get("u").copy(),
            x=current_particles.get("x").copy(),
            logdetj=current_particles.get("logdetj").copy(),
            logp=current_particles.get("logp").copy(),
            logl=current_particles.get("logl").copy(),
            beta=current_particles.get("beta"),
        )

        function_dict = dict(
            loglike=self._log_like,
            logprior=self.log_prior,
            scaler=self.scaler,
            flow=self.flow,
            u_geometry=self.u_geometry,
            theta_geometry=self.theta_geometry,
        )

        option_dict = dict(
            n_max=self.n_max_steps,
            n_steps=self.n_steps,
            progress_bar=self.pbar,
        )

        if self.preconditioned and self.sample == "pcn":
            results = preconditioned_pcn(
                state_dict,
                function_dict,
                option_dict
                )
        elif self.preconditioned and self.sample == "rwm":
            results = preconditioned_rwm(
                state_dict,
                function_dict,
                option_dict
                )
        elif not self.preconditioned and self.sample == "pcn":
            results = pcn(
                state_dict,
                function_dict,
                option_dict
                )
        elif not self.preconditioned and self.sample == "rwm":
            results = rwm(
                state_dict,
                function_dict,
                option_dict
                )

        current_particles["u"] = results.get('u').copy()
        current_particles["x"] = results.get('x').copy()
        current_particles["logdetj"] = results.get('logdetj').copy()
        current_particles["logl"] = results.get('logl').copy()
        current_particles["logp"] = results.get('logp').copy()
        current_particles["efficiency"] = results.get('efficiency') / (2.38 / self.n_dim ** 0.5)
        current_particles["steps"] = results.get('steps')
        current_particles["accept"] = results.get('accept')
        current_particles["calls"] = current_particles.get("calls") + results.get('calls')

        return current_particles


    def _train(self, current_particles):
        """
        Train normalizing flow.

        Parameters
        ----------
        current_particles : dict
            Dictionary containing the current particles.
        
        Returns
        -------
        current_particles : dict
            Dictionary containing the updated particles.
        """
        u = current_particles.get("u")
        w = current_particles.get("weights")

        if self.preconditioned:

            self.flow.fit(numpy_to_torch(u),
                          weights=numpy_to_torch(w),
                          validation_split=self.train_config["validation_split"],
                          epochs=self.train_config["epochs"],
                          batch_size=int(np.minimum(len(u)//2, self.train_config["batch_size"])),
                          gaussian_scale=self.train_config["gaussian_scale"],
                          laplace_scale=self.train_config["laplace_scale"],
                          patience=self.train_config["patience"],
                          learning_rate=self.train_config["learning_rate"],
                          annealing=self.train_config["annealing"],
                          noise=self.train_config["noise"],
                          shuffle=self.train_config["shuffle"],
                          clip_grad_norm=self.train_config["clip_grad_norm"],
                          verbose=self.train_config["verbose"],
                          )
            
            theta = flow_numpy_wrapper(self.flow).forward(u)[0]
            self.theta_geometry.fit(theta, weights=w)
        else:
            self.u_geometry.fit(u, weights=w)



        return current_particles

    def _resample(self, current_particles):
        """
        Resample particles.

        Parameters
        ----------
        current_particles : dict
            Dictionary containing the current particles.

        Returns
        -------
        current_particles : dict
            Dictionary containing the updated particles.
        """
        u = current_particles.get("u")
        x = current_particles.get("x")
        logdetj = current_particles.get("logdetj")
        logl = current_particles.get("logl")
        logp = current_particles.get("logp")
        weights = current_particles.get("weights")

        if self.resample == 'multinomial':
            idx_resampled = np.random.choice(np.arange(len(weights)), size=self.n_active, replace=True, p=weights)
        elif self.resample == 'systematic':
            idx_resampled = systematic_resample(self.n_active, weights=weights)

        current_particles["u"] = u[idx_resampled]
        current_particles["x"] = x[idx_resampled]
        current_particles["logdetj"] = logdetj[idx_resampled]
        current_particles["logl"] = logl[idx_resampled]
        current_particles["logp"] = logp[idx_resampled]
    
        return current_particles

    def _reweight(self, current_particles):
        """
        Reweight particles.

        Parameters
        ----------
        current_particles : dict
            Dictionary containing the current particles.

        Returns
        -------
        current_particles : dict
            Dictionary containing the updated particles.
        """
        # Update iteration index
        self.t += 1
        self.pbar.update_iter()

        beta_prev = self.particles.get("beta", index=-1)
        beta_max = 1.0
        beta_min = np.copy(beta_prev)

        def get_weights_and_ess_(beta):
            logw, _ = self.particles.compute_logw_and_logz(beta)
            weights = np.exp(logw - np.max(logw))
            weights /= np.sum(weights)
            ess_est = 1.0 / np.sum(weights**2.0)
            return weights, ess_est
        
        def get_weights_and_ess(beta):
            logw, _ = self.particles.compute_logw_and_logz(beta)
            weights = np.exp(logw - np.max(logw))
            weights /= np.sum(weights)
            #ess_est = 1.0 / np.sum(weights**2.0)
            expected_unique_all = np.sum(1-(1-weights)**len(weights))
            return weights, expected_unique_all

        weights_prev, ess_est_prev = get_weights_and_ess(beta_prev)
        weights_max, ess_est_max = get_weights_and_ess(beta_max)


        if ess_est_prev <= self.n_ess:
            beta = beta_prev
            weights = weights_prev
            logz = self.particles.get("logz", index=-1)
            self.pbar.update_stats(dict(beta=beta, ESS=ess_est_prev, logZ=logz))
        elif ess_est_max >= self.n_ess:
            beta = beta_max 
            weights = weights_max
            _, logz = self.particles.compute_logw_and_logz(beta)
            self.pbar.update_stats(dict(beta=beta, ESS=ess_est_max, logZ=logz))
        else:
            while True:
                beta = (beta_max + beta_min) * 0.5

                weights, ess_est = get_weights_and_ess(beta)

                if np.abs(ess_est - self.n_ess) < 0.01 * self.n_ess or beta == 1.0:
                    _, logz = self.particles.compute_logw_and_logz(beta)
                    self.pbar.update_stats(dict(beta=beta, ESS=ess_est, logZ=logz))
                    break
                elif ess_est < self.n_ess:
                    beta_max = beta
                else:
                    beta_min = beta

        logw, _ = self.particles.compute_logw_and_logz(beta)
        weights = np.exp(logw - np.max(logw))
        weights /= np.sum(weights)

        #expected_unique = np.sum(1-(1-weights)**self.n_active)
        #expected_unique_all = np.sum(1-(1-weights)**len(weights))
        #print("Expected unique particles: ", expected_unique)
        #print("Expected unique particles (all): ", expected_unique_all)
        if self.dynamic:
            n_unique_active = np.sum(1-(1-weights)**self.n_active)
            if n_unique_active < self.n_active * 0.75:
                self.n_ess = int(self.n_active/n_unique_active * self.n_ess)

        idx, weights = trim_weights(np.arange(len(weights)), weights, ess=0.99, bins=1000)
        current_particles["u"] = self.particles.get("u", index=None, flat=True)[idx]
        current_particles["x"] = self.particles.get("x", index=None, flat=True)[idx]
        current_particles["logdetj"] = self.particles.get("logdetj", index=None, flat=True)[idx]
        current_particles["logl"] = self.particles.get("logl", index=None, flat=True)[idx]
        current_particles["logp"] = self.particles.get("logp", index=None, flat=True)[idx]
        current_particles["logz"] = logz
        current_particles["beta"] = beta
        current_particles["weights"] = weights

        return current_particles

    def _log_like(self, x):
        """
        Compute log likelihood.

        Parameters
        ----------
        x : array_like
            Array of parameter values.
        
        Returns
        -------
        logl : float
            Log likelihood.
        """
        if self.vectorize_likelihood:
            return self.log_likelihood(x)
        elif self.pool is not None:
            return np.array(list(self.distribute(self.log_likelihood, x)))
        else:
            return np.array(list(map(self.log_likelihood, x)))
        
    def evidence(self):
        """
        Return the log evidence estimate and error.
        """
        return self.logz, self.logz_err
        
    def _compute_evidence(self, n=5_000):
        """
        Estimate the evidence using importance sampling.

        Parameters
        ----------
        n : int
            Number of importance samples (default is ``n=5_000``).
        
        Returns
        -------
        logz : float
            Estimate of the log evidence.
        dlogz : float
            Estimate of the error on the log evidence.
        """
        with torch.no_grad():
            theta_q, logq = self.flow.sample(n)
            theta_q = torch_to_numpy(theta_q)
            logq = torch_to_numpy(logq)

        x_q, logdetj = self.scaler.inverse(theta_q)
        logl = self._log_like(x_q)
        logp = self.log_prior(x_q)

        logw = logl + logp + logdetj - logq 
        logz = np.logaddexp.reduce(logw) - np.log(len(logw))

        dlogz = np.std([np.logaddexp.reduce(logw[np.random.choice(len(logw), len(logw))]) - np.log(len(logw)) for _ in range(np.maximum(n,1000))])

        self.logz = logz
        self.logz_err = dlogz
        return logz, dlogz

    def __getstate__(self):
        """
        Get state information for pickling.
        """
        state = self.__dict__.copy()

        try:
            # remove random module
            # del state['rstate']

            # deal with pool
            if state['pool'] is not None:
                del state['pool']  # remove pool
                del state['distribute']  # remove `pool.map` function hook
        except:  # TODO use specific exception type
            pass

        return state

    def posterior(self, resample=False, trim_importance_weights=True, return_logw=False, ess_trim=0.99, bins_trim=1_000):
        """
        Return posterior samples.

        Parameters
        ----------
        resample : bool
            If True, resample particles (default is ``resample=False``).
        trim_importance_weights : bool
            If True, trim importance weights (default is ``trim_importance_weights=True``).
        return_logw : bool
            If True, return log importance weights (default is ``return_logw=False``).
        ess_trim : float
            Effective sample size threshold for trimming (default is ``ess_trim=0.99``).
        bins_trim : int
            Number of bins for trimming (default is ``bins_trim=1_000``).

        Returns
        -------
        samples : ``np.ndarray``
            Samples from the posterior.
        weights : ``np.ndarray``
            Importance weights.
        logl : ``np.ndarray``
            Log likelihoods.
        logp : ``np.ndarray``
            Log priors.
        """
        samples = self.particles.get("x", flat=True)
        logl = self.particles.get("logl", flat=True)
        logp = self.particles.get("logp", flat=True)
        logw, _ = self.particles.compute_logw_and_logz(1.0)
        weights = np.exp(logw)

        if trim_importance_weights:
            idx, weights = trim_weights(np.arange(len(samples)), weights, ess=ess_trim, bins=bins_trim)
            samples = samples[idx]
            logl = logl[idx]
            logp = logp[idx]
            logw = logw[idx]

        if resample:
            if self.resample == 'multinomial':
                idx_resampled = np.random.choice(np.arange(len(weights)), size=len(samples), replace=True, p=weights)
            elif self.resample == 'systematic':
                idx_resampled = systematic_resample(len(weights), weights=weights)
            return samples[idx_resampled], logl[idx_resampled], logp[idx_resampled]
            
        else:
            if return_logw:
                return samples, logw, logl, logp
            else:
                return samples, weights, logl, logp

    @property
    def results(self):
        """
        Return results.

        Returns
        -------
        results : dict
            Dictionary containing the results.
        """
        return self.particles.compute_results()

    def save_state(self, path: Union[str, Path]):
        """Save current state of sampler to file.

        Parameters
        ----------
        path : ``Union[str, Path]``
            Path to save state.
        """
        print(f'Saving PMC state to {path}')
        Path(path).parent.mkdir(exist_ok=True)
        with open(path, 'wb') as f:
            state = self.__dict__.copy()
            del state['pbar']  # Cannot be pickled
            try:
                # deal with pool
                if state['pool'] is not None:
                    del state['pool']  # remove pool
                    del state['distribute']  # remove `pool.map` function hook
            except BaseException as e:
                print(e)
            dill.dump(file=f, obj=state)

    def load_state(self, path: Union[str, Path]):
        """Load state of sampler from file.

        Parameters
        ----------
        path : ``Union[str, Path]``
            Path from which to load state.
        """
        with open(path, 'rb') as f:
            state = dill.load(file=f)
        self.__dict__ = {**self.__dict__, **state}

