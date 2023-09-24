from typing import Any
import numpy as np

class Particles:

    def __init__(self, n_particles, n_dim, ess_threshold=None):
        self.n_particles = n_particles
        self.n_dim = n_dim
        self.ess_threshold = ess_threshold

        self.past = dict(
            u = [],
            x = [],
            logdetj = [],
            logl = [],
            logp = [],
            logw = [],
            iter = [],
            logz = [],
            calls = [],
            steps = [],
            efficiency = [],
            ess = [],
            accept = [],
            beta = [],
        )

        self.results_dict = None

    def update(self, data):
        for key in data.keys():
            if key in self.past.keys():
                value = data.get(key)
                # Save to past states
                self.past.get(key).append(value)

    def pop(self, key):
        _ = self.past.get(key).pop()

    def get(self, key, index=None, flat=False):
        if index is None:
            if flat:
                return np.concatenate(self.past.get(key))
            else:
                return np.asarray(self.past.get(key))
        else:
            return self.past.get(key)[index]
        
    def compute_logw(self, beta, ess_threshold=None):
        logl = self.get("logl")
        beta_original = self.get("beta")

        logw = (beta - beta_original[:,None]) * logl
        logw -= np.logaddexp.reduce(logw, axis=1)[:,None]
        log_ess = - np.logaddexp.reduce(2.0 * logw, axis=1)
        ###
        if ess_threshold is not None:
            mask = np.exp(log_ess) < ess_threshold
            log_ess[mask] = -1e300
        ###
        log_ess -= np.logaddexp.reduce(log_ess)
        logw += log_ess[:,None]
        logw = logw.reshape(-1)
        logw -= np.logaddexp.reduce(logw)
        
        return logw
    
    def compute_logz(self, beta):
        logz = self.get("logz")

        logl = self.get("logl")
        beta_original = self.get("beta")

        logw = (beta - beta_original[:,None]) * logl
        logw_normed = logw - np.logaddexp.reduce(logw, axis=1)[:,None]

        logz_increments = np.logaddexp.reduce(logw, axis=1) - np.log(logw.shape[1])
        
        log_ess = - np.logaddexp.reduce(2.0 * logw_normed, axis=1)
        log_ess_normed = log_ess - np.logaddexp.reduce(log_ess)

        logz_new = np.logaddexp.reduce(log_ess_normed + logz + logz_increments)

        return logz_new
    
    def compute_results(self):
        if self.results_dict is None:
            self.results_dict = dict()
            for key in self.past.keys():
                self.results_dict[key] = self.get(key)
        
            logl = self.get("logl")
            beta_original = self.get("beta")

            logw = (1.0 - beta_original[:,None]) * logl
            logw -= np.logaddexp.reduce(logw, axis=1)[:,None]
            log_ess = - np.logaddexp.reduce(2.0 * logw, axis=1)
            ###
            if self.ess_threshold is not None:
                mask = np.exp(log_ess) < self.ess_threshold
                log_ess[mask] = -1e300
            ###
            log_ess -= np.logaddexp.reduce(log_ess)
            logw += log_ess[:,None]

            self.results_dict["logw"] = logw
            self.results_dict["ess"] = np.exp(log_ess)

        return self.results_dict

