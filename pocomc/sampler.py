from pathlib import Path
from typing import Union

import os
import dill
import numpy as np
from multiprocess import Pool
import torch

from .mcmc import preconditioned_pcn, preconditioned_rwm, pcn, rwm
from .tools import systematic_resample, FunctionWrapper, numpy_to_torch, torch_to_numpy, trim_weights, ProgressBar, flow_numpy_wrapper, effective_sample_size, unique_sample_size
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
    n_effective : int
        The number of effective particles (default is ``n_effective=512``). Higher values
        lead to more accurate results but also increase the computational cost.  This should be
        set to a value that is large enough to ensure that the target distribution is well
        represented by the particles. The number of effective particles should be greater than
        the number of active particles. If ``n_effective=None``, the default value is ``n_effective=2*n_active``.
    n_active : int
        The number of active particles (default is ``n_active=256``). It must be smaller than ``n_effective``.
        For best results, the number of active particles should be no more than half the number of effective particles.
        This is the number of particles that are evolved using MCMC at each iteration. If a pool is provided,
        the number of active particles should be a multiple of the number of processes in the pool to ensure
        efficient parallelisation. If ``n_active=None``, the default value is ``n_active=n_effective//2``.
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
    blobs_dtype : list
        Data type of the blobs returned by the likelihood function (default is ``blobs_dtype=None``). If ``blobs_dtype``
        is not provided, the data type is inferred from the blobs returned by the likelihood function. If the blobs
        are not of the same data type, they are converted to an object array. If the blobs are strings, the data type
        is set to ``object``. If the blobs ``dtype`` is known in advance, it can be provided as a list of data types
        (e.g., ``blobs_dtype=[("blob_1", float), ("blob_2", int)]``). Blobs can be used to store additional data 
        returned by the likelihood function (e.g., chi-squared values, residuals, etc.). Blobs are stored as a
        structured array with named fields when the data type is provided. Currently, the blobs feature is not
        compatible with vectorized likelihood calculations.
    pool : pool or int
        Number of processes to use for parallelisation (default is ``pool=None``). If ``pool`` is an integer
        greater than 1, a ``multiprocessing`` pool is created with the specified number of processes (e.g., ``pool=8``). 
        If ``pool`` is an instance of ``mpi4py.futures.MPIPoolExecutor``, the code runs in parallel using MPI.
        If a pool is provided, the number of active particles should be a multiple of the number of processes in 
        the pool to ensure efficient parallelisation. If ``pool=None``, the code runs in serial mode. When a pool 
        is provided, please ensure that the likelihood function is picklable. 
    pytorch_threads : int
        Maximum number of threads to use for torch. If ``None`` torch uses all
        available threads while training the normalizing flow (default is ``pytorch_threads=1``). 
    flow : ``zuko.flow.Flow`` or str
        Normalizing flow to use for preconditioning (default is ``flow='nsf3'``). Available options are
        ``'nsf3'``, ``'nsf6'``, ``'nsf12'``, ``'maf3'``, ``'maf6'``, and ``'maf12'``. 'nsf' stands for
        Neural Spline Flows and 'maf' stands for Masked Autoregressive Flows. The number indicates the
        number of transformations in the flow. More transformations lead to more flexibility but also
        increase the computational cost. If a ``zuko.flow.Flow`` instance is provided, the normalizing 
        flow is used as is. If a string is provided, a new instance of the normalizing flow is created 
        with the specified architecture. The normalizing flow is used to precondition the MCMC sampler 
        and improve the efficiency of the sampling.
    train_config : dict or ``None``
        Configuration for training the normalizing flow
        (default is ``train_config=None``). Options include a dictionary with the following
        keys: ``"validation_split"``, ``"epochs"``, ``"batch_size"``, ``"patience"``,
        ``"learning_rate"``, ``"annealing"``, ``"gaussian_scale"``, ``"laplace_scale"``,
        ``"noise"``, ``"shuffle"``, ``"clip_grad_norm"``.
    train_frequency : int or None
        Frequency of training the normalizing flow (default is ``train_frequency=None``).
        If ``train_frequency=None``, the normalizing flow is trained every ``n_effective//n_active``
        iterations. If ``train_frequency=1``, the normalizing flow is trained at every iteration.
        If ``train_frequency>1``, the normalizing flow is trained every ``train_frequency`` iterations.
    precondition : bool
        If True, use preconditioned MCMC (default is ``precondition=True``). If False,
        use standard MCMC without normalizing flow. The use of preconditioned MCMC is
        recommended as it is more efficient and scales better with the number of parameters. 
        However, it requires the use of a normalizing flow and the training of the flow
        can be computationally expensive. If ``precondition=False``, the normalizing flow
        is not used and the sampler runs in standard mode. This works well for targets that
        are not multimodal or have strong non-linear correlations between parameters.
    dynamic : bool
        If True, dynamically adjust the effective sample size (ESS) threshold based on the
        number of unique particles (default is ``dynamic=False``). This can be useful for
        targets with a large number of modes or strong non-linear correlations between parameters.
    metric : str
        Metric used for determining the next temperature (``beta``) level (default is ``metric="ess"``).
        Options are ``"ess"`` (Effective Sample Size) or ``"uss"`` (Unique Sample Size). The metric
        is used to determine the next temperature level based on the ESS or USS of the importance
        weights. If the ESS or USS of the importance weights is below the target threshold, the temperature
        is increased. If the ESS or USS is above the target threshold, the temperature is decreased. The
        target threshold is set by the ``n_effective`` parameter.
    n_prior : int
        Number of prior samples to draw (default is ``n_prior=2*(n_effective//n_active)*n_active``). This
        is used to initialise the particles at the beginning of the run. The prior samples are used to
        warm-up the sampler and ensure that the particles are well distributed across the prior volume.
    sample : ``str``
        Type of MCMC sampler to use (default is ``sample="tpcn"``). Options are
        ``"pcn"`` (t-preconditioned Crank-Nicolson) or ``"rwm"`` (Random-walk Metropolis).
        t-preconditioned Crank-Nicolson is the default and recommended sampler for PMC as it
        is more efficient and scales better with the number of parameters.
    n_steps : int
        Number of MCMC steps after logP plateau (default is ``n_steps=n_dim``). This is used
        for early stopping of MCMC. Higher values can lead to better exploration but also
        increase the computational cost. If ``n_steps=None``, the default value is ``n_steps=n_dim``.
    n_max_steps : int
        Maximum number of MCMC steps (default is ``n_max_steps=10*n_dim``).
    resample : ``str``
        Resampling scheme to use (default is ``resample="mult"``). Options are
        ``"syst"`` (systematic resampling) or ``"mult"`` (multinomial resampling).
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
                 n_effective: int = 512,
                 n_active: int = 256,
                 likelihood_args: list = None,
                 likelihood_kwargs: dict = None,
                 vectorize: bool = False,
                 blobs_dtype: str = None,
                 pool=None,
                 pytorch_threads=1,
                 flow='nsf3',
                 train_config: dict = None,
                 train_frequency: int = None,
                 precondition: bool = True,
                 dynamic: bool = True,
                 metric: str = 'ess',
                 n_prior: int = None,
                 sample: str = 'tpcn',
                 n_steps: int = None,
                 n_max_steps: int = None,
                 resample: str = 'mult',
                 output_dir: str = None,
                 output_label: str = None,
                 random_state: int = None,
                 # deprecated
                 n_ess: int = None,
                 ):
        
        # Deprecation warnings
        if n_ess is not None:
            n_effective = n_ess
            # raise warning and print it but do not raise exception
            import warnings
            warnings.warn("n_ess is deprecated. Use n_effective instead.", DeprecationWarning, stacklevel=2)

        # Random seed
        if random_state is not None:
            np.random.seed(random_state)
            torch.manual_seed(random_state)
        self.random_state = random_state

        # Configure PyTorch threads
        configure_threads(pytorch_threads=pytorch_threads)

        # Prior distribution
        self.prior = prior
        self.log_prior = self.prior.logpdf
        self.sample_prior = self.prior.rvs
        self.bounds = self.prior.bounds

        # Log likelihood function
        self.log_likelihood = FunctionWrapper(
            likelihood,
            likelihood_args,
            likelihood_kwargs
        )

        # Blobs data type
        self.blobs_dtype = blobs_dtype
        self.have_blobs = blobs_dtype is not None

        # Number of parameters
        if n_dim is None:
            self.n_dim = self.prior.dim
        else:
            self.n_dim = int(n_dim)

        # Check that at least one parameter is provided
        if n_active is None and n_effective is None:
            raise ValueError("At least one of n_active or n_effective must be provided.")

        # Number of active particles
        if n_active is None:
            self.n_active = int(n_effective/2)
        else:
            self.n_active = int(n_active)

        # Effective Sample Size
        if n_effective is None:
            self.n_effective = int(2*n_active)
        else:
            self.n_effective = int(n_effective)

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

        # Number of samples for evidence estimation
        self.n_evidence = None

        # Particle manager
        self.particles = Particles(n_active, n_dim)

        # Iteration counter
        self.t = 0

        # Parallelism
        self.pool = pool
        if pool is None:
            self.distribute = map
        elif isinstance(pool, int) and pool > 1:
            self.pool = Pool(pool)
            self.distribute = self.pool.map
        else:
            self.distribute = pool.map

        # Likelihood vectorization
        self.vectorize = vectorize
        if self.vectorize and self.have_blobs:
            raise ValueError("Cannot vectorize likelihood with blobs.")

        # Geometry
        self.u_geometry = Geometry()
        self.theta_geometry = Geometry()

        # Normalizing Flow
        self.flow = Flow(self.n_dim, flow)
        self.train_config = dict(validation_split=0.5,
                                 epochs=5000,
                                 batch_size=np.minimum(self.n_effective//2, 512),
                                 patience=int(self.n_dim),
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
        
        if train_frequency is None:
            self.train_frequency = np.maximum(self.n_effective//(self.n_active*2), 1)
        else:
            self.train_frequency = int(train_frequency)

        self.flow_untrained = True

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

        # Normalizing flow preconditioning
        self.preconditioned = precondition

        # Effective vs Unique Sample Size
        if metric not in ['ess', 'uss']:
            raise ValueError(f"Invalid metric {metric}. Options are 'ess' or 'uss'.")
        else:
            self.metric = metric

        # Dynamic ESS
        self.dynamic = dynamic
        self.dynamic_ratio = unique_sample_size(np.ones(self.n_effective), k=self.n_active) / self.n_active

        # Sampling algorithm
        if sample not in ['tpcn', 'rwm']:
            raise ValueError(f"Invalid sample {sample}. Options are 'tpcn' or 'rwm'.")
        else:
            self.sample = sample

        # Proposal scale
        self.proposal_scale = 2.38 / self.n_dim ** 0.5

        # Resampling algorithm
        if resample not in ['mult', 'syst']:
            raise ValueError(f"Invalid resample {resample}. Options are 'mult' or 'syst'.")
        else:
            self.resample = resample

        # Prior samples to draw
        if n_prior is None:
            self.n_prior = int(2 * np.maximum(self.n_effective//self.n_active, 1) * self.n_active)
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
            n_total: int = 4096,
            n_evidence: int = 4096,
            progress: bool = True,
            resume_state_path: Union[str, Path] = None,
            save_every: int = None):
        r"""Run Preconditioned Monte Carlo.

        Parameters
        ----------
        n_total : int
            The total number of effectively independent samples to be
            collected (default is ``n_total=2048``).
        n_evidence : int
            The number of importance samples used to estimate the
            evidence (default is ``n_evidence=4096``). If ``n_evidence=0``,
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
            self.pbar = ProgressBar(self.progress, initial=t0)
            self.pbar.update_stats(dict(beta=self.particles.get("beta", -1),
                                        calls=self.particles.get("calls", -1),
                                        ESS=self.particles.get("ess", -1),
                                        logZ=self.particles.get("logz", -1),
                                        logP=np.mean(self.particles.get("logp", -1)+self.particles.get("logl", -1)),
                                        acc=self.particles.get("accept", -1),
                                        steps=self.particles.get("steps", -1),
                                        eff=self.particles.get("efficiency", -1)))
        else:
            t0 = self.t
            # Run parameters
            self.progress = progress

            # Initialise progress bar
            self.pbar = ProgressBar(self.progress)
            self.pbar.update_stats(dict(beta=0.0,
                                        calls=self.calls,
                                        ESS=self.n_effective,
                                        logZ=0.0,
                                        logP=0.0,
                                        acc=0.0,
                                        steps=0,
                                        eff=0.0))
            
        self.n_total = int(n_total)
        self.n_evidence = int(n_evidence)

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
                logl, blobs = self._log_like(x)
                self.calls += self.n_active

                self.current_particles = dict(u=u,x=x,logl=logl,logp=logp,logdetj=logdetj,
                                    logw=-1e300 * np.ones(self.n_active), blobs=blobs, iter=self.t,
                                    calls=self.calls, steps=1, efficiency=1.0, ess=self.n_effective, 
                                    accept=1.0, beta=0.0, logz=0.0)
                
                self.particles.update(self.current_particles)

                self.pbar.update_stats(dict(calls=self.particles.get("calls", -1), 
                                            beta=self.particles.get("beta", -1), 
                                            ESS=int(self.particles.get("ess", -1)),
                                            logZ=self.particles.get("logz", -1),
                                            logP=np.mean(self.particles.get("logp", -1)+self.particles.get("logl", -1)),
                                            acc=self.particles.get("accept", -1),
                                            steps=self.particles.get("steps", -1),
                                            eff=self.particles.get("efficiency", -1)))
                
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
        if self.n_evidence > 0 and self.preconditioned:
            self._compute_evidence(self.n_evidence)
        else:
            _, self.logz = self.particles.compute_logw_and_logz(1.0)
            self.logz_err = None
        
        # Save final state
        if save_every is not None:
            self.save_state(Path(self.output_dir) / f'{self.output_label}_final.state')
        
        # Close progress bar
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
        if self.metric == 'ess':
            ess = effective_sample_size(weights)
        elif self.metric == 'uss':
            ess = unique_sample_size(weights)

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
        if self.have_blobs:
            blobs = current_particles.get("blobs").copy()
        else:
            blobs = None
        state_dict = dict(
            u=current_particles.get("u").copy(),
            x=current_particles.get("x").copy(),
            logdetj=current_particles.get("logdetj").copy(),
            logp=current_particles.get("logp").copy(),
            logl=current_particles.get("logl").copy(),
            beta=current_particles.get("beta"),
            blobs=blobs,
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
            proposal_scale=self.proposal_scale,
        )

        if self.preconditioned and self.sample == "tpcn":
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
        elif not self.preconditioned and self.sample == "tpcn":
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
        if self.have_blobs:
            current_particles["blobs"] = results.get('blobs').copy()
        current_particles["efficiency"] = results.get('efficiency') / (2.38 / self.n_dim ** 0.5)
        current_particles["steps"] = results.get('steps')
        current_particles["accept"] = results.get('accept')
        current_particles["calls"] = current_particles.get("calls") + results.get('calls')
        self.calls = current_particles.get("calls")
        self.proposal_scale = results.get('proposal_scale')

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

        if self.preconditioned and (self.t % self.train_frequency == 0 or current_particles.get("beta")==1.0 or self.flow_untrained):
            self.flow_untrained = False
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
        blobs = current_particles.get("blobs")

        if self.resample == 'mult':
            idx_resampled = np.random.choice(np.arange(len(weights)), size=self.n_active, replace=True, p=weights)
        elif self.resample == 'syst':
            idx_resampled = systematic_resample(self.n_active, weights=weights)

        current_particles["u"] = u[idx_resampled]
        current_particles["x"] = x[idx_resampled]
        current_particles["logdetj"] = logdetj[idx_resampled]
        current_particles["logl"] = logl[idx_resampled]
        current_particles["logp"] = logp[idx_resampled]
        if self.have_blobs:
            current_particles["blobs"] = blobs[idx_resampled]
    
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

        def get_weights_and_ess(beta):
            logw, _ = self.particles.compute_logw_and_logz(beta)
            weights = np.exp(logw - np.max(logw))
            if self.metric == 'ess':
                ess_est = effective_sample_size(weights)
            elif self.metric == 'uss':
                ess_est = unique_sample_size(weights)
            return weights, ess_est

        weights_prev, ess_est_prev = get_weights_and_ess(beta_prev)
        weights_max, ess_est_max = get_weights_and_ess(beta_max)


        if ess_est_prev <= self.n_effective:
            beta = beta_prev
            weights = weights_prev
            logz = self.particles.get("logz", index=-1)
            ess_est = ess_est_prev
            self.pbar.update_stats(dict(beta=beta, ESS=int(ess_est_prev), logZ=logz))
        elif ess_est_max >= self.n_effective:
            beta = beta_max 
            weights = weights_max
            _, logz = self.particles.compute_logw_and_logz(beta)
            ess_est = ess_est_max
            self.pbar.update_stats(dict(beta=beta, ESS=int(ess_est_max), logZ=logz))
        else:
            while True:
                beta = (beta_max + beta_min) * 0.5

                weights, ess_est = get_weights_and_ess(beta)

                if np.abs(ess_est - self.n_effective) < 0.01 * self.n_effective or beta == 1.0:
                    _, logz = self.particles.compute_logw_and_logz(beta)
                    self.pbar.update_stats(dict(beta=beta, ESS=int(ess_est), logZ=logz))
                    break
                elif ess_est < self.n_effective:
                    beta_max = beta
                else:
                    beta_min = beta

        logw, _ = self.particles.compute_logw_and_logz(beta)
        weights = np.exp(logw - np.max(logw))
        weights /= np.sum(weights)

        if self.dynamic:
            # Adjust the number of effective particles based on the expected number of unique particles
            n_unique_active = unique_sample_size(weights, k=self.n_active)
            # Maintain the original ratio of unique active to effective particles
            if n_unique_active < self.n_active * (0.95 * self.dynamic_ratio):
                self.n_effective = int(self.n_active/n_unique_active * self.n_effective)
            elif n_unique_active > self.n_active * np.minimum(1.05 * self.dynamic_ratio, 1.0):
                self.n_effective = int(n_unique_active/self.n_active * self.n_effective)

        idx, weights = trim_weights(np.arange(len(weights)), weights, ess=0.99, bins=1000)
        current_particles["u"] = self.particles.get("u", index=None, flat=True)[idx]
        current_particles["x"] = self.particles.get("x", index=None, flat=True)[idx]
        current_particles["logdetj"] = self.particles.get("logdetj", index=None, flat=True)[idx]
        current_particles["logl"] = self.particles.get("logl", index=None, flat=True)[idx]
        current_particles["logp"] = self.particles.get("logp", index=None, flat=True)[idx]
        if self.have_blobs:
            current_particles["blobs"] = self.particles.get("blobs", index=None, flat=True)[idx]
        current_particles["logz"] = logz
        current_particles["beta"] = beta
        current_particles["weights"] = weights
        current_particles["ess"] = ess_est

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
        blob : array_like
            Additional data (default is ``None``).
        """
        if self.vectorize:
            return self.log_likelihood(x), None
        elif self.pool is not None:
            results = list(self.distribute(self.log_likelihood, x))
        else:
            results = list(map(self.log_likelihood, x))


        try:
            blob = [l[1:] for l in results if len(l) > 1]
            if not len(blob):
                raise IndexError
            logl = np.array([float(l[0]) for l in results])
            self.have_blobs = True
        except (IndexError, TypeError):
            logl = np.array([float(l) for l in results])
            blob = None
        else:
            # Get the blobs dtype
            if self.blobs_dtype is not None:
                dt = self.blobs_dtype
            else:
                try:
                    dt = np.atleast_1d(blob[0]).dtype
                except ValueError:
                    dt = np.dtype("object")
                if dt.kind in "US":
                    # Strings need to be object arrays or we risk truncation
                    dt = np.dtype("object")
            blob = np.array(blob, dtype=dt)

            # Deal with single blobs properly
            shape = blob.shape[1:]
            if len(shape):
                axes = np.arange(len(shape))[np.array(shape) == 1] + 1
                if len(axes):
                    blob = np.squeeze(blob, tuple(axes))

        return logl, blob
        
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
        logl, _ = self._log_like(x_q)
        logp = self.log_prior(x_q)

        logw = logl + logp + logdetj - logq 
        logz = np.logaddexp.reduce(logw) - np.log(len(logw))

        dlogz = np.std([np.logaddexp.reduce(logw[np.random.choice(len(logw), len(logw))]) - np.log(len(logw)) for _ in range(np.maximum(n,1000))])

        self.calls += n
        self.pbar.update_stats(dict(calls=self.calls))

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

    def posterior(self, resample=False, return_blobs=False, trim_importance_weights=True, return_logw=False, ess_trim=0.99, bins_trim=1_000):
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
        if return_blobs and not self.have_blobs:
            raise ValueError("No blobs available.")

        samples = self.particles.get("x", flat=True)
        logl = self.particles.get("logl", flat=True)
        logp = self.particles.get("logp", flat=True)
        if return_blobs:
            blobs = self.particles.get("blobs", flat=True)
        logw, _ = self.particles.compute_logw_and_logz(1.0)
        weights = np.exp(logw)

        if trim_importance_weights:
            idx, weights = trim_weights(np.arange(len(samples)), weights, ess=ess_trim, bins=bins_trim)
            samples = samples[idx]
            logl = logl[idx]
            logp = logp[idx]
            logw = logw[idx]

        if resample:
            if self.resample == 'mult':
                idx_resampled = np.random.choice(np.arange(len(weights)), size=len(samples), replace=True, p=weights)
            elif self.resample == 'syst':
                idx_resampled = systematic_resample(len(weights), weights=weights)
            if return_blobs:
                return samples[idx_resampled], logl[idx_resampled], logp[idx_resampled], blobs[idx_resampled]
            else:
                return samples[idx_resampled], logl[idx_resampled], logp[idx_resampled]
            
        else:
            if return_logw:
                if return_blobs:
                    return samples, logw, logl, logp, blobs
                else:
                    return samples, logw, logl, logp
            else:
                if return_blobs:
                    return samples, weights, logl, logp, blobs
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
        temp_path = Path(path).with_suffix('.temp')
        with open(temp_path, 'wb') as f:
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
            f.flush()
            os.fsync(f.fileno())

        os.rename(temp_path, path)

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

