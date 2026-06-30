from scipy.optimize import brentq
import numpy as np
from scipy.stats import norm

class EuropeanAnalyticsSABR:
    from scipy.optimize import brentq

    def __init__(self, F0, T, beta, nu, rho, sigma_atm, lambda_):
        self.F0 = F0
        self.T = T
        self.beta = beta
        self.nu = nu
        self.rho = rho
        self.sigma_atm = sigma_atm
        self.lambda_ = lambda_
        self.F_s = F0 + lambda_
        
        # Solve for alpha such that hagan_implied_vol(F0) == sigma_atm
        def objective(alpha_trial):
            self.alpha = alpha_trial
            return self.hagan_implied_vol(F0) - sigma_atm
        
        # Just using a fixed upper bound
        upper = 10.0
        self.alpha = brentq(objective, 1e-8, upper, xtol=1e-12)


    def hagan_implied_vol(self, K):
        K_s = K + self.lambda_
        F_s = self.F_s
        alpha = self.alpha
        beta = self.beta
        nu = self.nu
        rho = self.rho
        T = self.T
        
        # ATM case when K=F
        if abs(F_s - K_s) < 1e-8 * F_s:
            correction = (1-beta)**2 * alpha**2 / (24 * F_s**(2-2*beta)) + rho*beta*nu*alpha / (4 * F_s**(1-beta)) + (2 - 3*rho**2)/24 * nu**2
            sigma_B = (alpha / F_s**(1-beta)) * (1 + correction * T)
            return sigma_B
        
        #General case when 
        FK_beta2 = (F_s * K_s) ** ((1.0 - beta) / 2.0)
        log_FK = np.log(F_s / K_s)

        # z and chi(z)
        z = nu / alpha * FK_beta2 * log_FK
        x_z = np.log((np.sqrt(1 - 2 * rho * z + z ** 2) + z - rho) / (1 - rho))

        denom = 1 + ((1-beta) ** 2 / 24 * log_FK ** 2) + ((1-beta) ** 4 / 1920 * log_FK ** 4)
        correction = (1-beta)**2 * alpha**2 / (24 * FK_beta2**2) + rho*beta*nu*alpha / (4 * FK_beta2) + (2 - 3*rho**2)/24 * nu**2

        sigma_B = (alpha / (FK_beta2 * denom)) * (z / x_z) * (1 + correction * T)
        return sigma_B
    
    def implied_vol_smile(self, strikes):
        return np.array([self.hagan_implied_vol(K) for K in strikes])

    def implied_vol_smile_mc(self, F_s_T, strikes):
        return np.array([self.bs_implied_vol(
            np.mean(np.maximum(F_s_T - (K + self.lambda_), 0)),
            self.F_s, 
            K + self.lambda_, 
            self.T) for K in strikes])

    @staticmethod
    def black_scholes_price(F, K, T, sigma, r=0, option_type="call"):
        if T <= 0 or sigma <= 0:
            intrinsic = max(F - K, 0) if option_type == "call" else max(K - F, 0)
            return intrinsic * np.exp(-r * T)
 
        d1 = (np.log(F / K) + 0.5 * sigma ** 2 * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        df = np.exp(-r * T)
 
        if option_type == "call":
            return df * (F * norm.cdf(d1) - K * norm.cdf(d2))
        else:
            return df * (K * norm.cdf(-d2) - F * norm.cdf(-d1))
        
    @staticmethod
    def bachelier_price(F, K, T, sigma, r=0, option_type="call"):
        if T <= 0 or sigma <= 0:
            intrinsic = max(F - K, 0) if option_type == "call" else max(K - F, 0)
            return intrinsic * np.exp(-r * T)
 
        d = (F - K) / (sigma * np.sqrt(T))
        df = np.exp(-r * T)
 
        if option_type == "call":
            return df * ((F - K) * norm.cdf(d) + sigma * np.sqrt(T) * norm.pdf(d))
        else:
            return df * ((K - F) * norm.cdf(-d) + sigma * np.sqrt(T) * norm.pdf(d))
        
    @staticmethod
    def bs_implied_vol(price, F, K, T, r=0, option_type="call", tol=1e-8):
        intrinsic = max(F - K, 0) if option_type == "call" else max(K - F, 0)
        if price <= intrinsic * np.exp(-r * T) + tol:
            return np.nan
        def objective(sigma):
            return EuropeanAnalyticsSABR.black_scholes_price(
                F, K, T, sigma, r, option_type) - price
        try:
            return brentq(objective, 1e-6, 10.0, xtol=tol, maxiter=200)
        except ValueError:
            return np.nan
 
    @staticmethod
    def bachelier_implied_vol(price, F, K, T, r=0, option_type="call", tol=1e-8):
        intrinsic = max(F - K, 0) if option_type == "call" else max(K - F, 0)
        if price <= intrinsic * np.exp(-r * T) + tol:
            return np.nan
        def objective(sigma_n):
            return EuropeanAnalyticsSABR.bachelier_price(
                F, K, T, sigma_n, r, option_type) - price
        try:
            return brentq(objective, 1e-8, 10.0, xtol=tol, maxiter=200)
        except ValueError:
            return np.nan

    def price(self, K, r=0, option_type="call"):
        sigma_B = self.hagan_implied_vol(K)
        K_s = K + self.lambda_
        return self.black_scholes_price(self.F_s, K_s, self.T, sigma_B, r, option_type)
    
    @staticmethod
    def mc_price(F_paths, K, T, r=0, option_type="call"):
        F_T = F_paths[:, -1]
        if option_type == "call":
            payoffs = np.maximum(F_T - K, 0.0)
        else:
            payoffs = np.maximum(K - F_T, 0.0)
 
        df = np.exp(-r * T)
        discounted = df * payoffs
        price = discounted.mean()
        stderr = discounted.std() / np.sqrt(len(discounted))
        return {
            "price":    price,
            "std_error": stderr,
            "ci_low":   price - 1.96 * stderr,
            "ci_high":  price + 1.96 * stderr,
        }