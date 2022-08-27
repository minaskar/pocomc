from pathlib import Path
from typing import Union

import dill
import numpy as np
import torch
from scipy.special import logsumexp
from scipy.optimize import root_scalar

from .input_validation import assert_array_2d
from .mcmc import preconditioned_metropolis, metropolis
from .tools import resample_equal, FunctionWrapper, numpy_to_torch, torch_to_numpy, compute_ess, increment_logz, ProgressBar
from .scaler import Reparameterise
from .flow import Flow


class Sampler:
    r"""Preconditioned Monte Carlo class.

    Parameters
    ----------
    n_particles : int
        The total number of particles/walkers to use.
    n_dim : int
        The total number of parameters/dimensions.
    log_likelihood : callable
        Function returning the log likelihood of a set
        of parameters.
    log_prior : callable
        Function returning the log prior of a set
        of parameters.
    bounds : ``np.ndarray`` or None
        Array of shape ``(ndim, 2)`` holding the boundaries
        of parameters (default is ``bounds=None``). If a
        parameter is unbounded from below, above or both
        please provide ``None`` for the respective boundary.
    periodic : list
        List of indices that correspond to parameters with
        periodic boundary conditions.
    reflective : list
        List of indices that correspond to parameters with
        reflective boundary conditions.
    transform : ``str``
        Type of transform to use for bounded parameters. Options are ``"probit"``
        (default) and ``"logit"``.
    threshold : float
        The threshold value for the (normalised) proposal
        scale parameter below which normalising flow
        preconditioning (NFP) is enabled (default is
        ``threshold=1.0``, meaning that NFP is used all
        the time).
    scale : bool
        Whether to scale the distribution of particles to
        have zero mean and unit variance. Default is ``True``.
    rescale : bool
        Whether to rescale the distribution of particles to
        have zero mean and unit variance in every iteration.
        Default is ``False``.
    diagonal : bool
        Use a diagonal covariance matrix when rescaling instead
        of a full covariance. Default is ``True``.
    log_likelihood_args : list
        Extra arguments to be passed to log_likelihood
        (default is ``loglikelihood_args=None``).
    log_likelihood_kwargs : dict
        Extra arguments to be passed to log_likelihood
        (default is ``loglikelihood_kwargs=None``).
    log_prior_args : list
        Extra arguments to be passed to log_prior
        (default is ``log_prior_args=None``).
    log_prior_kwargs : list
        Extra arguments to be passed to log_prior
        (default is ``log_prior_kwargs=None``).
    vectorize_likelihood : bool
        If True, vectorize ``loglikelihood``
        calculation (default is ``vectorize_likelihood=False``).
    vectorize_prior : bool
        If True, vectorize ``log_prior``
        calculation (default is ``vectorize_prior=False``).
    infer_vectorization : bool
        If True, infer the vectorization status of
        the loglikelihood and logprior automatically. Default
        is ``True`` (overwrites the ``vectorize_likelihood``
        and ``vectorize_prior`` arguments).
    pool : pool
        Provided ``MPI`` or ``multiprocessing`` pool for
        parallelisation (default is ``pool=None``).
    parallelize_prior : bool
        If True, use the ``pool`` (if provided)
        for the ``logprior`` as well (default is
        ``parallelize_prior=False``).
    flow_config : dict or ``None``
        Configuration of the normalizing flow (default is
        ``flow_config=None``).
    train_config : dict or ``None``
        Configuration for training the normalizing flow
        (default is ``train_config=None``).
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
                 n_particles: int,
                 n_dim: int,
                 log_likelihood: callable,
                 log_prior: callable,
                 bounds: np.ndarray = None,
                 periodic=None,
                 reflective=None,
                 transform="probit",
                 threshold: float = 1.0,
                 scale: bool = True,
                 rescale: bool = False,
                 diagonal: bool = True,
                 log_likelihood_args: list = None,
                 log_likelihood_kwargs: dict = None,
                 log_prior_args: list = None,
                 log_prior_kwargs: dict = None,
                 vectorize_likelihood: bool = False,
                 vectorize_prior: bool = False,
                 infer_vectorization: bool = True,
                 pool=None,
                 parallelize_prior: bool = False,
                 flow_config: dict = None,
                 train_config: dict = None,
                 output_dir: str = None,
                 output_label: str = None,
                 random_state: int = None):

        if random_state is not None:
            np.random.seed(random_state)
            torch.manual_seed(random_state)
        self.random_state = random_state

        self.n_walkers = n_particles
        self.n_dim = n_dim

        # Distributions
        self.log_likelihood = FunctionWrapper(
            log_likelihood,
            log_likelihood_args,
            log_likelihood_kwargs
        )
        self.log_prior = FunctionWrapper(
            log_prior,
            log_prior_args,
            log_prior_kwargs
        )

        # Sampling
        self.n_call = 0
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
        self.saved_n_call = []
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
        self.infer_vectorization = infer_vectorization
        self.parallelize_prior = parallelize_prior

        # Flow
        self.flow = Flow(self.n_dim, flow_config, train_config)
        self.threshold = threshold
        self.use_flow = False

        # Scaler
        self.scaler = Reparameterise(
            n_dim=self.n_dim,
            bounds=bounds,
            periodic=periodic,
            reflective=reflective,
            transform=transform,
            scale=scale,
            diagonal=diagonal
        )
        self.rescale = rescale

        # MCMC parameters
        self.ideal_scale = 2.38 / np.sqrt(n_dim)
        self.scale = 2.38 / np.sqrt(n_dim)
        self.accept = 0.234
        self.target_accept = 0.234

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
        self.ess = None
        self.gamma = None
        self.n_min = None
        self.n_max = None
        self.progress = None
        self.pbar = None

    def validate_vectorization_settings(self, x_check: np.ndarray):
        """
        Check that vectorization settings are sensible.
        This involves making a likelihood and prior call for each sample in x_check to determine the output shapes.
        User-provided vectorization settings are overwritten according to output shapes.
        The user is warned about the changes.
        If both outputs are arrays (neither is a scalar) of different shape, an error is raised.

        Parameters
        ----------
        x_check: np.ndarray
            Input array used to check vectorization settings.
        """

        def is_function_vectorized(f, n_test=2):
            try:
                output_multiple = f(x_check[:n_test, :])
                if output_multiple.shape == (n_test,):
                    return True
                else:
                    raise ValueError
            except (ValueError, AttributeError):
                output_single = f(x_check[0])
                if isinstance(output_single, float):
                    return False
                else:
                    raise ValueError

        # Use three test samples if ndim = 2
        self.vectorize_likelihood = is_function_vectorized(self.log_likelihood, 3 if self.n_dim == 2 else 2)
        self.vectorize_prior = is_function_vectorized(self.log_prior, 3 if self.n_dim == 2 else 2)

    def run(self,
            prior_samples: np.ndarray = None,
            ess: float = 0.95,
            gamma: float = 0.75,
            n_min: int = None,
            n_max: int = None,
            progress: bool = True,
            resume_state_path: Union[str, Path] = None,
            save_every: int = None):
        r"""Run Preconditioned Monte Carlo.

        Parameters
        ----------
        prior_samples : ``np.ndarray``
            Array holding the initial positions of the particles. The initial
            positions must be sampled from the prior distribution.
        ess : float
            The effective sample size maintained during the run (default is
            `ess=0.95`).
        gamma : float
            Threshold for the correlation coefficient that is
            used to adaptively determine the number of MCMC
            steps (default is ``gamma=0.75``).
        n_min : int or None
            The minimum number of MCMC steps per iteration (default is ``n_min = ndim // 2``).
        n_max : int or None
            The maximum number of MCMC steps per iteration  (default is ``n_min = int(10 * n_dim)``).
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
        else:
            if prior_samples is None:
                raise ValueError(
                    f"Prior samples not provided. You can only omit these if you load an existing PMC state by "
                    f"specifying load_state_path."
                )

            assert_array_2d(prior_samples)
            if self.infer_vectorization:
                self.validate_vectorization_settings(prior_samples)

                # Run parameters
            self.ess = ess
            self.gamma = gamma
            if n_min is None:
                self.n_min = self.n_dim // 2
            else:
                self.n_min = int(n_min)
            if n_max is None:
                self.n_max = int(10 * self.n_dim)
            else:
                self.n_max = int(n_max)
            self.progress = progress

            # Set state parameters
            self.x = np.copy(prior_samples)
            self.scaler.fit(self.x)
            self.u = self.scaler.forward(self.x)
            self.J = self.scaler.inverse(self.u)[1]
            self.P = self._log_prior(self.x)
            finite_prior_mask = np.isfinite(self.P)
            self.L = np.full((len(self.x),), -np.inf)
            self.L[finite_prior_mask] = self._log_like(self.x[finite_prior_mask])
            self.n_call += sum(finite_prior_mask)

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
            self.saved_logw.append(np.zeros(self.n_walkers))
            self.saved_logz.append(self.logz)
            self.saved_ess.append(self.ess)
            self.saved_n_call.append(self.n_call)
            self.saved_accept.append(self.accept)
            self.saved_scale.append(self.scale / self.ideal_scale)
            self.saved_steps.append(0)

        # Initialise progress bar
        self.pbar = ProgressBar(self.progress)
        self.pbar.update_stats(
            dict(
                beta=self.beta,
                calls=self.n_call,
                ESS=self.ess,
                logZ=self.logz,
                accept=0.234,
                N=0,
                scale=1.0
            )
        )

        t0 = self.t
        # Run Sequential Monte Carlo
        while 1.0 - self.beta >= 1e-4:
            if save_every is not None:
                if (self.t - t0) % int(save_every) == 0 and self.t != t0:
                    self.save_state(Path(self.output_dir) / f'{self.output_label}_{self.t}.state')

            # Choose next beta based on CV of weights
            self._update_beta()

            # Resample x_prev, w
            self.u, self.x, self.J, self.L, self.P = self._resample(
                self.u,
                self.x,
                self.J,
                self.L,
                self.P
            )

            # Evolve particles using MCMC
            self.u, self.x, self.J, self.L, self.P = self._mutate(
                self.u,
                self.x,
                self.J,
                self.L,
                self.P
            )

            # Rescale parameters
            if self.rescale:
                self.scaler.fit(self.x)
                self.u = self.scaler.forward(self.x)
                self.J = self.scaler.inverse(self.u)[1]

            # Train Preconditioner
            self._train(self.u)

        self.saved_posterior_samples.append(self.x.copy())
        self.saved_posterior_logl.append(self.L.copy())
        self.saved_posterior_logp.append(self.P.copy())

        self.pbar.close()

    def add_samples(self,
                    n: int = 1000,
                    retrain: bool = False,
                    progress: bool = True):
        r"""Method that generates additional samples at the end of the run

        Parameters
        ----------
        n : int
            The number of additional samples. Default: ``1000``.
        retrain : bool
            If True, retrain the normalising flow preconditioner between iterations. Default: ``False``.
        progress : bool
            If True, show progress bar. Default: ``True``.
        """
        self.progress = progress

        self.pbar = ProgressBar(self.progress)
        self.pbar.update_stats(
            dict(
                beta=self.beta,
                calls=self.n_call,
                ESS=self.ess,
                logZ=self.logz,
                accept=0,
                N=0,
                scale=0
            )
        )

        iterations = int(np.ceil(n / len(self.u)))
        for _ in range(iterations):
            self.u, self.x, self.J, self.L, self.P = self._mutate(
                self.u,
                self.x,
                self.J,
                self.L,
                self.P
            )
            if retrain:
                self._train(self.u)
            self.saved_posterior_samples.append(self.x)
            self.saved_posterior_logl.append(self.L)
            self.saved_posterior_logp.append(self.P)
            self.pbar.update_iter()

        self.pbar.close()
        # self.u = np.tile(self.u.T, multiply).T

    def _mutate(self, u, x, J, L, P):
        """
            Method which mutates particle positions

        Parameters
        ----------
        u : ``np.ndarray``
            Scaled positions of particles.
        x : ``np.ndarray``
            Unscaled (original) positions of particles.
        J : ``np.ndarray``
            Logarithms of the absolute determinant of the Jacobian of the scaling transform.
        L : ``np.ndarray``
            log-likelihood values of particles
        P : ``np.ndarray``
            log-prior values of particles

        Returns
        -------
        u : ``np.ndarray``
            Mutated scaled positions of particles.
        x : ``np.ndarray``
            Mutated unscaled (original) positions of particles.
        J : ``np.ndarray``
            Mutated logarithms of the absolute determinant of the Jacobian of the scaling transform.
        L : ``np.ndarray``
            Mutated log-likelihood values of particles
        P : ``np.ndarray``
            Mutated log-prior values of particles
        """
        state_dict = dict(
            u=u.copy(),
            x=x.copy(),
            J=J.copy(),
            L=L.copy(),
            P=P.copy(),
            beta=self.beta
        )

        function_dict = dict(
            loglike=self._log_like,
            logprior=self._log_prior,
            scaler=self.scaler,
            flow=self.flow
        )

        option_dict = dict(
            nmin=self.n_min,
            nmax=self.n_max,
            corr_threshold=self.gamma,
            sigma=self.scale,
            progress_bar=self.pbar
        )

        if self.use_flow:
            results = preconditioned_metropolis(
                state_dict,
                function_dict,
                option_dict
            )
        else:
            results = metropolis(
                state_dict,
                function_dict,
                option_dict
            )

        u = results.get('u').copy()
        x = results.get('x').copy()
        J = results.get('J').copy()
        L = results.get('L').copy()
        P = results.get('P').copy()

        self.scale = results.get('scale')
        n_steps = results.get('steps')
        self.accept = results.get('accept')

        self.n_call += results.get('calls')

        self.saved_n_call.append(self.n_call)
        self.saved_accept.append(self.accept)
        self.saved_scale.append(self.scale / self.ideal_scale)
        self.saved_steps.append(n_steps)

        return u, x, J, L, P

    def _train(self, u):
        """
            Method which trains the normalising flow.

        Parameters
        ----------
        u : ``np.ndarray``
            Input training data (i.e. positions of particles)
        """
        if (self.scale < self.threshold * self.ideal_scale and self.t > 1) or self.use_flow:
            y = np.copy(u)
            np.random.shuffle(y)
            self.flow.fit(numpy_to_torch(y))
            self.use_flow = True
        else:
            pass

    def _resample(self, u, x, J, L, P):
        """
            Method which resamples particle positions

        Parameters
        ----------
        u : ``np.ndarray``
            Scaled positions of particles.
        x : ``np.ndarray``
            Unscaled (original) positions of particles.
        J : ``np.ndarray``
            Logarithms of the absolute determinant of the Jacobian of the scaling transform.
        L : ``np.ndarray``
            log-likelihood values of particles
        P : ``np.ndarray``
            log-prior values of particles

        Returns
        -------
        u : ``np.ndarray``
            Resampled scaled positions of particles.
        x : ``np.ndarray``
            Resampled unscaled (original) positions of particles.
        J : ``np.ndarray``
            Resampled logarithms of the absolute determinant of the Jacobian of the scaling transform.
        L : ``np.ndarray``
            Resampled log-likelihood values of particles
        P : ``np.ndarray``
            Resampled log-prior values of particles
        """
        self.saved_samples.append(x)
        self.saved_logl.append(L)
        self.saved_logp.append(P)
        w = np.exp(self.logw - np.max(self.logw))
        w /= np.sum(w)

        assert np.all(~np.isnan(self.logw))
        assert np.all(np.isfinite(self.logw))
        assert np.all(~np.isnan(w))
        assert np.all(np.isfinite(w))

        try:
            idx = resample_equal(np.arange(len(u)), w, self.random_state)
        except IndexError:
            #warnings.warn("Systematic resampling failed. Trying multinomial resampling.")
            idx = np.random.choice(np.arange(len(u)), p=w, size=len(w))
        self.logw = 0.0

        return u[idx], x[idx], J[idx], L[idx], P[idx]

    def _update_beta(self):
        """
            Update beta level and evidence estimate.
        """
        # Update iteration index
        self.t += 1
        self.saved_iter.append(self.t)
        self.pbar.update_iter()

        beta_prev = np.copy(self.beta)
        beta_max = 1.0
        beta_min = np.copy(beta_prev)
        logw_prev = np.copy(self.logw)

        while True:
            beta = (beta_max + beta_min) * 0.5
            self.logw = logw_prev + self.L * (beta - beta_prev)
            ess_est = compute_ess(self.logw)

            if len(self.saved_beta) > 1:
                dbeta = self.saved_beta[-1] - self.saved_beta[-2]

                if 1.0 - beta < dbeta * 0.1:
                    beta = 1.0

            if np.abs(ess_est - self.ess) < min(0.001 * self.ess, 0.001) or beta == 1.0:
                self.saved_beta.append(beta)
                self.saved_logw.append(self.logw)
                self.sum_logw += self.logw
                self.saved_ess.append(ess_est)
                self.beta = beta
                self.pbar.update_stats(dict(beta=self.beta, ESS=ess_est))
                # Update evidence 
                self.logz += increment_logz(self.logw)
                self.saved_logz.append(self.logz)
                self.pbar.update_stats(dict(logZ=self.logz))
                break

            elif ess_est < self.ess:
                beta_max = beta
            else:
                beta_min = beta

    def _log_prior(self, x):
        """
            Compute the log-prior values of the particles.

        Parameters
        ----------
        x : ``np.ndarray``
            Input array of particle positions.

        Returns
        -------
        P : ``np.ndarray``
            Array of log-prior values of particles.
        """
        if self.vectorize_prior:
            return self.log_prior(x)
        elif self.parallelize_prior and self.pool is not None:
            return np.array(list(self.distribute(self.log_prior, x)))
        else:
            return np.array(list(map(self.log_prior, x)))

    def _log_like(self, x):
        """
            Compute the log-likelihood values of the particles.

        Parameters
        ----------
        x : ``np.ndarray``
            Input array of particle positions.

        Returns
        -------
        L : ``np.ndarray``
            Array of log-likelihood values of particles.
        """
        if self.vectorize_likelihood:
            return self.log_likelihood(x)
        elif self.pool is not None:
            return np.array(list(self.distribute(self.log_likelihood, x)))
        else:
            return np.array(list(map(self.log_likelihood, x)))

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

    @property
    def results(self):
        """
            Results dictionary. Includes the following properties: 
            ``samples``, ``loglikelihood``, ``logprior``, ``iter``
            ``logw``, ``logl``, ``logp``, ``logz``, ``ess``, ``ncall``,
            ``beta``, ``accept``, ``scale``, and ``steps``.
        """
        return {
            'samples': np.vstack(self.saved_posterior_samples),
            'loglikelihood': np.hstack(self.saved_posterior_logl),
            'logprior': np.hstack(self.saved_posterior_logp),
            'logz': np.array(self.saved_logz),
            'iter': np.array(self.saved_iter),
            'x': np.array(self.saved_samples),
            'logl': np.array(self.saved_logl),
            'logp': np.array(self.saved_logp),
            'logw': np.array(self.saved_logw),
            'ess': np.array(self.saved_ess),
            'ncall': np.array(self.saved_n_call),
            'beta': np.array(self.saved_beta),
            'accept': np.array(self.saved_accept),
            'scale': np.array(self.saved_scale),
            'steps': np.array(self.saved_steps)
        }

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

    def bridge_sampling(self, maxiter=2000, rtol=1e-13, xtol=1e-13, thin=1):
        r"""
            Bridge Sampling estimator for the log model evidence.
            This implements the Gaussianised Bridge Sampling 
            algorithm https://arxiv.org/abs/1912.06073 which
            utilises a normalising flow as the auxiliary distribution.

        Parameters
        ----------
        maxiter : ``int``
            Maximum number of iterations of root-finding procedure.
        rtol : ``float``
            Relative numerical tolerance of root-finding procedure.
        xtol : ``float``
            Absolute numerical tolerance of root-finding procedure.
        thin : ``int``
            Thin the samples by a integer factor. Default is ``thin=1``
            (no thinning).
        Returns
        -------
        logr : ``float``
            Estimate of log model evidence.
        logr_err : ``float``
            Estimate of the 1-sigma uncertainty of the log model evidence.
        """

        x = self.results.get("samples")[::thin]
        l = self.results.get("loglikelihood")[::thin]
        p = self.results.get("logprior")[::thin]
        ess = self.ess

        N1 = len(x)
        N2 = len(x)

        s1 = N1 / (N1+N2)
        s2 = N2 / (N1+N2)

        theta_prop = torch.randn((N2, self.n_dim))
        u_prop, log_abs_det_jac = self.flow.inverse(theta_prop)
        logg_i = torch.sum(self.flow.flow.base_dist.log_prob(theta_prop) - log_abs_det_jac, dim=1)
        u_prop = torch_to_numpy(u_prop)
        x_prop, J_prop = self.scaler.inverse(u_prop)
        logg_i = torch_to_numpy(logg_i) - J_prop

        u = self.scaler.forward(x)
        x, J = self.scaler.inverse(u)
        logg_j = torch_to_numpy(self.flow.logprob(numpy_to_torch(u))) - J

        log_prior_tmp = self._log_prior(x_prop)
        finite_prior_mask = np.isfinite(log_prior_tmp)
        log_like_tmp = np.full((len(x_prop), ), -np.inf)
        log_like_tmp[finite_prior_mask] = self._log_like(x_prop[finite_prior_mask])
        logp_i = log_prior_tmp + log_like_tmp
        logp_j = p + l

        _a = logg_j - logp_j - np.log(N1 / N2)
        _b = logp_i - logg_i + np.log(N1 / N2)

        def score(logr):
            _c = logsumexp(logr + _a - logsumexp(np.array((logr + _a,
                        np.zeros_like(_a))), axis=0))
            _d = logsumexp(-logr + _b - logsumexp(np.array((-logr + _b,
                        np.zeros_like(_b))), axis=0))
            return _c - _d

        logr = root_scalar(score, x0=0, x1=5, maxiter=maxiter, rtol=rtol, xtol=xtol).root

        # Uncertainty
        f1 = np.exp(logp_i - logr - logsumexp(np.array((logp_i - logr +
                    np.log(s1), logg_i + np.log(s2))), axis=0))
        f2 = np.exp(logg_j - logsumexp(np.array((logp_j - logr +
                    np.log(s1), logg_j + np.log(s2))), axis=0))
        re2_q = np.var(f1) / np.mean(f1)**2 / N2

        tau = 1.0 / ess
        re2_p = tau * np.var(f2) / np.mean(f2)**2 / N1
        logr_err = (re2_p + re2_q)**0.5

        return logr, logr_err
