from stoch_vol import StochasticVolSimulator
import numpy as np

class SABR(StochasticVolSimulator):
    def __init__(self, F0, A0, rho, T, n_steps, n_paths, beta, nu, seed=None):
        super().__init__(F0, A0, rho, T, n_steps, n_paths, seed)
        self.beta = beta
        self.nu = nu

    def local_vol(self, F):
        return F**self.beta

    def vol_drift(self, A):
        return np.zeros_like(A)

    def vol_of_vol(self, A):
        return self.nu * A
    

class mSABR(StochasticVolSimulator):
    def __init__(self, F0, A0, rho, T, n_steps, n_paths, beta, nu, kappa, theta, seed=None):
        super().__init__(F0, A0, rho, T, n_steps, n_paths, seed)
        self.beta = beta
        self.nu = nu
        self.kappa = kappa
        self.theta = theta

    def local_vol(self, F):
        return F**self.beta

    def vol_drift(self, A):
        return self.kappa * (self.theta - A)

    def vol_of_vol(self, A):
        return self.nu * A