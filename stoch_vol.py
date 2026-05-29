from abc import ABC, abstractmethod
import numpy as np
from stats import PathStatistics

class StochasticVolSimulator(ABC):
    def __init__(self, F0, A0, rho, T, n_steps, n_paths, seed=None):
        self.F0 = F0
        self.A0 = A0
        self.rho = rho
        self.T = T
        self.n_steps = n_steps
        self.n_paths = n_paths
        self.rng = np.random.default_rng(seed)

    @abstractmethod
    def local_vol(self, F):
        pass

    @abstractmethod
    def vol_drift(self, A):
        pass

    @abstractmethod
    def vol_of_vol(self, A):
        pass
    
    def compute_stats(self, paths):
        stats = {
            "mean": PathStatistics.mean(paths),
            "variance": PathStatistics.variance(paths),
            "skewness": PathStatistics.skewness(paths),
            "kurtosis": PathStatistics.kurtosis(paths),
            "quantile_25": PathStatistics.quantile(paths, 25),
            "quantile_50": PathStatistics.quantile(paths, 50),
            "quantile_75": PathStatistics.quantile(paths, 75),
            "max": PathStatistics.max(paths),
            "min": PathStatistics.min(paths),
            "std_dev": PathStatistics.std_dev(paths),
        }
        return stats

    def run(self):
        dt = self.T / self.n_steps
        F = np.zeros((self.n_paths, self.n_steps + 1))
        A = np.zeros((self.n_paths, self.n_steps + 1))
        F[:, 0] = self.F0
        A[:, 0] = self.A0

        for i in range(self.n_steps):
            Z1 = self.rng.standard_normal(self.n_paths)
            Z2 = self.rng.standard_normal(self.n_paths)
            dW1 = np.sqrt(dt) * Z1
            dW2 = np.sqrt(dt) * (self.rho * Z1 + np.sqrt(1 - self.rho**2) * Z2)
            F[:, i+1] = F[:, i] + A[:, i] * self.local_vol(F[:, i]) * dW1
            F[:, i+1] = np.maximum(F[:, i+1], 0.0)
            A[:, i+1] = A[:, i] + self.vol_drift(A[:, i]) * dt + self.vol_of_vol(A[:, i]) * dW2
            A[:, i+1] = np.maximum(A[:, i+1], 0.0)
        return F, A