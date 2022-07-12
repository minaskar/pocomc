from abc import abstractmethod, ABC
from typing import Union

import numpy as np
import torch
import torch.distributions as D


class Distribution(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def sample(self, n: int) -> np.ndarray:
        pass

    @abstractmethod
    def log_prob(self, x: np.ndarray) -> np.array:
        pass

    @property
    def bounds(self):
        return np.nan, np.nan


class Uniform(Distribution):
    def __init__(self,
                 lower: Union[np.ndarray, float],
                 upper: Union[np.ndarray, float],
                 n_dim: int = None):
        super().__init__()
        if not isinstance(lower, np.ndarray) and not isinstance(upper, np.ndarray) and n_dim is None:
            raise ValueError(f"n_dim must be provided when lower and upper bounds are floats.")
        self.n_dim = n_dim
        self.lower = lower
        self.upper = upper
        if not isinstance(lower, np.ndarray):
            self.lower = np.array([lower for _ in range(n_dim)], dtype=np.float32)
        else:
            self.lower = np.asarray(lower, dtype=np.float32)

        if not isinstance(upper, np.ndarray):
            self.upper = np.array([upper for _ in range(n_dim)], dtype=np.float32)
        else:
            self.upper = np.asarray(upper, dtype=np.float32)
        self.dist = D.Uniform(low=torch.as_tensor(self.lower), high=torch.as_tensor(self.upper))
        self.log_prob_value = -np.sum(np.log(self.upper - self.lower))

    @torch.no_grad()
    def log_prob(self, x):
        if len(x.shape) == 1:
            x = x.reshape(1, self.n_dim)
        return np.where(np.all((x >= self.lower) & (x <= self.upper), axis=1), self.log_prob_value, -np.inf)

    @torch.no_grad()
    def sample(self, n: int):
        return self.dist.sample((n,)).numpy()

    @property
    def bounds(self):
        return np.c_[self.lower, self.upper]
