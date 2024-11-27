from pathlib import Path
from typing import Union

import os
import dill
import numpy as np
from multiprocess import Pool

from .mcmc import parallel_mcmc
from .tools import systematic_resample, FunctionWrapper, trim_weights, ProgressBar, effective_sample_size, unique_sample_size
from .particles import Particles
from .cluster import RecursiveDensityClustering
from .student import fit_mvstud

class Sampler:
    r"""Persistent Sampling class.

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
    periodic : list or ``None``
        List of parameter indeces that should be wrapped around the domain (default is ``periodic=None``).
        This can be useful for phase parameters that might be periodic e.g. on a range ``[0,2*np.pi]``. For example,
        ``periodic=[0,1]`` will wrap around the first and second parameters.
    reflective : list or ``None``
        List of parameter indeces that should be reflected around the domain (default is ``reflective=None``).
        This can arise in cases where parameters are ratios where ``a/b`` and  ``b/a`` are equivalent. For example,
        ``reflective=[0,1]`` will reflect the first and second parameters.
    transform : str
        Type of transformation to apply to bounded parameters (default is ``transform='probit'``). Available options
        are ``'probit'`` and ``'logit'``. See ``Reparameterize`` for more details.
    pool : pool or int
        Number of processes to use for parallelisation (default is ``pool=None``). If ``pool`` is an integer
        greater than 1, a ``multiprocessing`` pool is created with the specified number of processes (e.g., ``pool=8``). 
        If ``pool`` is an instance of ``mpi4py.futures.MPIPoolExecutor``, the code runs in parallel using MPI.
        If a pool is provided, the number of active particles should be a multiple of the number of processes in 
        the pool to ensure efficient parallelisation. If ``pool=None``, the code runs in serial mode. When a pool 
        is provided, please ensure that the likelihood function is picklable. 
    train_frequency : int or None
        Frequency of training the normalizing flow (default is ``train_frequency=None``).
        If ``train_frequency=None``, the normalizing flow is trained every ``n_effective//n_active``
        iterations. If ``train_frequency=1``, the normalizing flow is trained at every iteration.
        If ``train_frequency>1``, the normalizing flow is trained every ``train_frequency`` iterations.
    dynamic : bool
        If True, dynamically adjust the effective sample size (ESS) threshold based on the
        number of unique particles (default is ``dynamic=True``). This can be useful for
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
                 prior_transform: callable,
                 log_likelihood: callable,
                 n_dim: int,
                 n_effective: int = 512,
                 n_active: int = 256,
                 log_likelihood_args: list = None,
                 log_likelihood_kwargs: dict = None,
                 vectorize: bool = False,
                 blobs_dtype: str = None,
                 periodic: list = None,
                 reflective: list = None,
                 dynamic: bool = True,
                 pool=None,
                 clustering: bool = True,
                 n_max_clusters: int = None,
                 metric: str = 'ess',
                 n_prior: int = None,
                 sample: str = 'tpcn',
                 n_steps: int = None,
                 n_max_steps: int = None,
                 resample: str = 'mult',
                 output_dir: str = None,
                 output_label: str = None,
                 random_state: int = None,
                 ):
        
        # Random seed
        if random_state is not None:
            np.random.seed(random_state)
        self.random_state = random_state

        # Prior distribution
        self.prior_transform = prior_transform

        # Log likelihood function
        self.log_likelihood = FunctionWrapper(
            log_likelihood,
            log_likelihood_args,
            log_likelihood_kwargs
        )

        # Blobs data type
        self.blobs_dtype = blobs_dtype
        self.have_blobs = blobs_dtype is not None

        # Number of parameters
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

        # Particle manager
        self.particles = Particles(n_active, n_dim)

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

        # Output
        if output_dir is None:
            self.output_dir = Path("states")
        else:
            self.output_dir = output_dir
        if output_label is None:
            self.output_label = "pmc"
        else:
            self.output_label = output_label

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

        # Clusterer
        self.clustering = clustering
        if self.clustering:
            self.clusterer = RecursiveDensityClustering(max_components=n_max_clusters,
                                                        n_init=10,
                                                        min_points=None,
                                                        alpha=2.0,
                                                        rescale=True,
                                                        verbose=False)
        else:
            self.clusterer = None

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

        self.warmup = True
        
        self.progress = None
        self.pbar = None

        # Particle Ensemble State
        self.u = None
        self.x = None
        self.logl = None
        self.assignments = None
        self.weights = None
        self.blobs = None
        self.acceptance = None
        self.steps = None
        self.efficiency = None
        self.ess = None
        self.beta = None
        self.logz = None
        self.calls = None
        self.iter = None


    def run(self,
            n_total: int = 4096,
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
            t0 = self.iter
            # Initialise progress bar
            self.pbar = ProgressBar(self.progress, initial=t0)
            self.pbar.update_stats(dict(beta=self.particles.get("beta", -1),
                                        calls=self.particles.get("calls", -1),
                                        ESS=self.particles.get("ess", -1),
                                        logZ=self.particles.get("logz", -1),
                                        logL=np.mean(self.particles.get("logl", -1)),
                                        acc=self.particles.get("accept", -1),
                                        steps=self.particles.get("steps", -1),
                                        eff=self.particles.get("efficiency", -1)))
        else:
            t0 = 0
            self.iter = 0
            self.calls = 0
            # Run parameters
            self.progress = progress

            # Initialise progress bar
            self.pbar = ProgressBar(self.progress)
            self.pbar.update_stats(dict(beta=0.0,
                                        calls=self.calls,
                                        ESS=self.n_effective,
                                        logZ=0.0,
                                        logL=0.0,
                                        acc=0.0,
                                        steps=0,
                                        eff=0.0))
            
        self.n_total = int(n_total)

        # Prior sampling
        if self.warmup:
            for i in range(self.n_prior//self.n_active):
                if save_every is not None:
                    if (self.iter - t0) % int(save_every) == 0 and self.iter != t0:
                        self.save_state(Path(self.output_dir) / f'{self.output_label}_{self.iter}.state')
                # Set state parameters
                self.u = np.random.rand(self.n_active, self.n_dim)
                self.x = np.array([self.prior_transform(self.u[i]) for i in range(self.n_active)])
                self.logl, self.blobs = self._log_like(self.x)
                self.assignments = np.zeros(self.n_active, dtype=int)
                self.calls += self.n_active
                self.steps = 1
                self.acceptance = 1.0
                self.efficiency = 1.0
                self.ess = self.n_effective
                self.beta = 0.0
                self.logz = 0.0

                # Resample prior particles with infinite likelihoods
                inf_logl_mask = np.isinf(self.logl)
                if np.any(inf_logl_mask):
                    all_idx = np.arange(len(self.x))
                    infinite_idx = all_idx[inf_logl_mask]
                    finite_idx = all_idx[~inf_logl_mask]
                    idx = np.random.choice(finite_idx, size=len(infinite_idx), replace=True)
                    self.x[infinite_idx] = self.x[idx]
                    self.u[infinite_idx] = self.u[idx]
                    self.logl[infinite_idx] = self.logl[idx]
                    if self.have_blobs:
                        self.blobs[infinite_idx] = self.blobs[idx]
                
                # Save particles
                self.particles.update({
                    "u" : self.u,
                    "x" : self.x,
                    "logl" : self.logl,
                    "assignments" : self.assignments,
                    "blobs" : self.blobs,
                    "iter" : self.iter,
                    "calls" : self.calls,
                    "steps" : self.steps,
                    "efficiency" : self.efficiency,
                    "ess" : self.ess,
                    "accept" : self.acceptance,
                    "beta" : self.beta,
                    "logz" : self.logz,
                })

                # Update progress bar
                self.pbar.update_stats(dict(calls=self.calls, 
                                            beta=self.beta, 
                                            ESS=self.ess,
                                            logZ=self.logz,
                                            logL=np.mean(self.logl),
                                            acc=self.acceptance,
                                            steps=self.steps,
                                            eff=self.efficiency))
                
                self.pbar.update_iter()

                self.iter += 1
            self.warmup = False

        # Run PS loop
        while self._not_termination():
            if save_every is not None:
                if (self.iter - t0) % int(save_every) == 0 and self.iter != t0:
                    self.save_state(Path(self.output_dir) / f'{self.output_label}_{self.iter}.state')

            # Choose next beta based on ESS of weights
            self._reweight()

            # Train clustering
            self._train()

            # Resample particles
            self._resample()

            # Evolve particles using MCMC
            self._mutate()   

            # Save particles
            self.particles.update({
                    "u" : self.u,
                    "x" : self.x,
                    "logl" : self.logl,
                    "assignments" : self.assignments,
                    "blobs" : self.blobs,
                    "iter" : self.iter,
                    "calls" : self.calls,
                    "steps" : self.steps,
                    "efficiency" : self.efficiency,
                    "ess" : self.ess,
                    "accept" : self.acceptance,
                    "beta" : self.beta,
                    "logz" : self.logz,
                })

        # Compute evidence
        _, self.logz = self.particles.compute_logw_and_logz(1.0)
        self.logz_err = None
        
        # Save final state
        if save_every is not None:
            self.save_state(Path(self.output_dir) / f'{self.output_label}_final.state')
        
        # Close progress bar
        self.pbar.close()

    def _not_termination(self):
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

        return 1.0 - self.beta >= 1e-4 or ess < self.n_total

    
    def _mutate(self):
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
            blobs = self.blobs.copy()
        else:
            blobs = None

        self.u, self.x, self.logl, blobs, self.efficiency, self.accept, self.steps, calls = parallel_mcmc(
            u=self.u,
            x=self.x,
            logl=self.logl,
            blobs=blobs,
            assignments=self.assignments,
            beta=self.beta,
            means=self.means,
            covariances=self.covariances,
            degrees_of_freedom=self.degrees_of_freedom,
            log_likelihood=self._log_like,
            prior_transform=self.prior_transform,
            progress_bar=self.pbar,
            n_steps=self.n_steps,
            n_max= self.n_max_steps,
            verbose=True,)

        if self.have_blobs:
            self.blobs = blobs.copy()
        self.calls += calls
    

    def _train(self):
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
        if self.clustering:
            u_resampled = self.u[np.random.choice(np.arange(len(self.weights)), size=self.n_effective*4, replace=True, p=self.weights)]

            self.clusterer.fit(u_resampled)
            labels = self.clusterer.predict(self.u)
            means = []
            covariances = []
            degrees_of_freedom = []
            for label in range(np.unique(labels).shape[0]):
                idx = np.where(labels == label)[0]
                mean, covariance, dof = fit_mvstud(self.u[idx])
                if ~np.isfinite(dof):
                    dof = 1e6 
                means.append(mean)
                covariances.append(covariance)
                degrees_of_freedom.append(dof)

            self.means = np.array(means)
            self.covariances = np.array(covariances)
            self.degrees_of_freedom = np.array(degrees_of_freedom)

            import matplotlib.pyplot as plt
            plt.scatter(self.u[:,0], self.u[:,1], c=labels)
            plt.show()
        else:
            self.means = None
            self.covariances = np.cov(self.u, rowvar=False, aweights=self.weights).reshape(1, self.n_dim, self.n_dim)
            self.degrees_of_freedom = None

    def _resample(self):
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
        u = self.u
        x = self.x
        logl = self.logl
        weights = self.weights
        blobs = self.blobs

        if self.resample == 'mult':
            idx_resampled = np.random.choice(np.arange(len(weights)), size=self.n_active, replace=True, p=weights)
        elif self.resample == 'syst':
            idx_resampled = systematic_resample(self.n_active, weights=weights)

        self.u = u[idx_resampled]
        self.x = x[idx_resampled]
        self.logl = logl[idx_resampled]
        if self.have_blobs:
            self.blobs = blobs[idx_resampled]

        if self.clustering:
            self.assignments = self.clusterer.predict(self.u)
        else:
            self.assignments = np.zeros(self.n_active, dtype=int)
    
    def _reweight(self):
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
        self.iter += 1
        self.pbar.update_iter()

        beta_prev = self.beta
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
            logz = self.logz
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
        self.u = self.particles.get("u", index=None, flat=True)[idx]
        self.x = self.particles.get("x", index=None, flat=True)[idx]
        self.logl = self.particles.get("logl", index=None, flat=True)[idx]
        if self.have_blobs:
            self.blobs = self.particles.get("blobs", index=None, flat=True)[idx]
        self.logz = logz
        self.beta = beta
        self.weights = weights
        self.ess = ess_est

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
        if return_blobs:
            blobs = self.particles.get("blobs", flat=True)
        logw, _ = self.particles.compute_logw_and_logz(1.0)
        weights = np.exp(logw)

        if trim_importance_weights:
            idx, weights = trim_weights(np.arange(len(samples)), weights, ess=ess_trim, bins=bins_trim)
            samples = samples[idx]
            logl = logl[idx]
            logw = logw[idx]
            if return_blobs:
                blobs = blobs[idx]

        if resample:
            if self.resample == 'mult':
                idx_resampled = np.random.choice(np.arange(len(weights)), size=len(samples), replace=True, p=weights)
            elif self.resample == 'syst':
                idx_resampled = systematic_resample(len(weights), weights=weights)
            if return_blobs:
                return samples[idx_resampled], logl[idx_resampled], blobs[idx_resampled]
            else:
                return samples[idx_resampled], logl[idx_resampled]
            
        else:
            if return_logw:
                if return_blobs:
                    return samples, logw, logl, blobs
                else:
                    return samples, logw, logl
            else:
                if return_blobs:
                    return samples, weights, logl, blobs
                else:
                    return samples, weights, logl

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

