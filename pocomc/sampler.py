import warnings

import numpy as np
import torch

from .input_validation import assert_array_2d, assert_arrays_equal_shape, assert_array_1d
from .mcmc import PreconditionedMetropolis, Metropolis
from .tools import resample_equal, FunctionWrapper, torch_to_numpy, numpy_to_torch, compute_ess, ProgressBar
from .scaler import Reparameterise
from .flow import Flow


class Sampler:
    r""" 

        A Preconditioned Monte Carlo class.

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
    periodic : list
        List of indeces that correspond to parameters with
        periodic boundary conditions.
    reflective : list
        List of indeces that correspond to parameters with
        reflective boundary conditions.
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
        have zero mean and unit variance in every iterations.
        Default is ``False``.
    diagonal : bool
        Use a diagonal covariance matrix when rescaling instead
        of a full covariance. Default is ``True``.
    loglikelihood_args : list
        Extra arguments to be passed into the loglikelihood
        (default is ``loglikelihood_args=None``).
    loglikelihood_kwargs : dict
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
    infer_vectorization : bool
        Whether or not to infer the vectorization status of
        the loglikelihood and logprior automatically. Default
        is ``True`` (overwrites the ``vectorize_likelihood``
        and ``vectorize_prior`` arguments).
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
    random_state : int or ``None``
        Initial random seed.
    """
    def __init__(self,
                 nparticles: int,
                 ndim: int,
                 loglikelihood: callable,
                 logprior: callable,
                 bounds: np.ndarray = None,
                 periodic=None,
                 reflective=None,
                 threshold: float = 1.0,
                 scale: bool = True,
                 rescale: bool = False,
                 diagonal: bool = True,
                 loglikelihood_args: list = None,
                 loglikelihood_kwargs: dict = None,
                 logprior_args: list = None,
                 logprior_kwargs: dict = None,
                 vectorize_likelihood: bool = False,
                 vectorize_prior: bool = False,
                 infer_vectorization: bool = True,
                 pool=None,
                 parallelize_prior: bool = False,
                 flow_config: dict = None,
                 train_config: dict = None,
                 random_state: int = None):
        
        if random_state is not None:
            np.random.seed(random_state)
            torch.manual_seed(random_state)
        self.random_state = random_state

        self.nwalkers = nparticles
        self.ndim = ndim

        # Distributions
        self.loglikelihood = FunctionWrapper(
            loglikelihood,
            loglikelihood_args,
            loglikelihood_kwargs
        )
        self.logprior = FunctionWrapper(
            logprior,
            logprior_args,
            logprior_kwargs
        )

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
        self.infer_vectorization = infer_vectorization
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

        # Other
        self.ess = None
        self.gamma = None
        self.nmin = None
        self.nmax = None
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
        def is_function_vectorized(f):
            try:
                output_multiple = f(x_check[:2, :])
                if output_multiple.shape == (2,):
                    return True
                else:
                    raise ValueError
            except ValueError:
                output_single = f(x_check[0])
                if isinstance(output_single, float):
                    return False
                else:
                    raise ValueError

        self.vectorize_likelihood = is_function_vectorized(self.loglikelihood)
        self.vectorize_prior = is_function_vectorized(self.logprior)

    def run(self,
            prior_samples: np.ndarray,
            ess: float = 0.95,
            gamma: float = 0.75,
            nmin: int = None,
            nmax: int = None,
            progress: bool = True):
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
        nmin : int or None
            The minimum number of MCMC steps per iteration (default is ``nmin = ndim // 2``).
        nmax : int or None
            The maximum number of MCMC steps per iteration  (default is ``nmin = int(10 * ndim)``).
        progress : bool
            Whether or not to print progress bar (default is ``progress=True``).
        """
        assert_array_2d(prior_samples)
        if self.infer_vectorization:
            self.validate_vectorization_settings(prior_samples)  

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
        self.x = np.copy(prior_samples)
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
        self.saved_scale.append(self.scale / self.ideal_scale)
        self.saved_steps.append(0)

        # Initialise progress bar
        self.pbar = ProgressBar(self.progress)
        self.pbar.update_stats(
            dict(
                beta=self.beta,
                calls=self.ncall,
                ESS=self.ess,
                logZ=self.logz,
                accept=0.234,
                N=0,
                scale=1.0
            )
        )

        # Run Sequential Monte Carlo
        while 1.0 - self.beta >= 1e-4:

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
                    N: int = 1000,
                    retrain: bool = False,
                    progress: bool = True):
        r"""Method that generates additional samples at the end of the run

        Parameters
        ----------
        N : int
            The number of additional samples. Default: ``1000``.
        retrain : bool
            Whether or not to retrain the normalising flow preconditioner between iterations. Default: ``False``.
        progress : bool
            Whether or not to show progress bar. Default: ``True``.
        """
        self.progress = progress

        self.pbar = ProgressBar(self.progress)
        self.pbar.update_stats(
            dict(
                beta=self.beta,
                calls=self.ncall,
                ESS=self.ess,
                logZ=self.logz,
                accept=0,
                N=0,
                scale=0
            )
        )

        iterations = int(np.ceil(N / len(self.u)))
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
            loglike=self._loglike,
            logprior=self._logprior,
            scaler=self.scaler,
            flow=self.flow
        )

        option_dict = dict(
            nmin=self.nmin,
            nmax=self.nmax,
            corr_threshold=self.gamma,
            sigma=self.scale,
            progress_bar=self.pbar
        )

        if self.use_flow:
            results = PreconditionedMetropolis(
                state_dict,
                function_dict,
                option_dict
            )
        else:
            results = Metropolis(
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
        self.nsteps = results.get('steps')
        self.accept = results.get('accept')

        self.ncall += self.nsteps * len(x)

        self.saved_ncall.append(self.ncall)
        self.saved_accept.append(self.accept)
        self.saved_scale.append(self.scale / self.ideal_scale)
        self.saved_steps.append(self.nsteps)

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
        except:  # TODO use specific exception type
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
                self.logz += np.mean(self.logw)
                self.saved_logz.append(self.logz)
                self.pbar.update_stats(dict(logZ=self.logz))
                break

            elif ess_est < self.ess:
                beta_max = beta
            else:
                beta_min = beta

    def _logprior(self, x):
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
            return self.logprior(x)
        elif self.parallelize_prior and self.pool is not None:
            return np.array(list(self.distribute(self.logprior, x)))
        else:
            return np.array(list(map(self.logprior, x)))

    def _loglike(self, x):
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
            return self.loglikelihood(x)
        elif self.pool is not None:
            return np.array(list(self.distribute(self.loglikelihood, x)))
        else:
            return np.array(list(map(self.loglikelihood, x)))

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
        results = {
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
            'ncall': np.array(self.saved_ncall),
            'beta': np.array(self.saved_beta),
            'accept': np.array(self.saved_accept),
            'scale': np.array(self.saved_scale),
            'steps': np.array(self.saved_steps)
        }

        return results

    def _bridge_sampling(self,
                        tolerance: float = 1e-10,
                        maxiter: int = 1000,
                        thin: int = 1):
        x = self.results.get("samples")[::thin]
        l = self.results.get("loglikelihood")[::thin]
        p = self.results.get("logprior")[::thin]

        N1 = len(x)
        N2 = len(x)

        s1 = N1 / (N1 + N2)
        s2 = N2 / (N1 + N2)

        u_prop, logg_i = self.flow.sample(size=N2)
        x_prop, J_prop = self.scaler.inverse(torch_to_numpy(u_prop))
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
        while np.abs(r - r0) > tolerance or cnt <= maxiter:
            r0 = r
            A = np.mean(l2i / (s1 * l2i + s2 * r0))
            B = np.mean(1.0 / (s1 * l1j + s2 * r0))
            r = A / B
            cnt += 1

        return np.log(r) + lstar
