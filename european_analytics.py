from scipy.optimize import brentq
from scipy.integrate import quad
import numpy as np
from scipy.stats import norm


class EuropeanAnalyticsSABR:

    def __init__(self, F_s, T, beta, nu, rho, sigma_atm, lambda_):
        self.T = T
        self.beta = beta
        self.nu = nu
        self.rho = rho
        self.sigma_atm = sigma_atm
        self.lambda_ = lambda_
        self.F_s = F_s
        
        atm_price = self.bachelier_price(self.F_s, self.F_s, self.T, self.sigma_atm)
        self.sigma_ln_atm = self.bs_implied_vol(atm_price, self.F_s, self.F_s, self.T)

        # Solve for alpha such that hagan_implied_vol(F_s) == sigma_ln_atm
        def objective(alpha_trial):
            self.alpha = alpha_trial
            return self.hagan_implied_vol(F_s) - self.sigma_ln_atm
        
        # Just using a fixed upper bound
        upper = 10.0
        self.alpha = brentq(objective, 1e-8, upper, xtol=1e-12)


    def hagan_implied_vol(self, K_s):
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

        # General case
        FK_beta2 = (F_s * K_s) ** ((1.0 - beta) / 2.0)
        log_FK = np.log(F_s / K_s)

        # z and chi(z)
        z = nu / alpha * FK_beta2 * log_FK
        x_z = np.log((np.sqrt(1 - 2 * rho * z + z ** 2) + z - rho) / (1 - rho))

        denom = 1 + ((1-beta) ** 2 / 24 * log_FK ** 2) + ((1-beta) ** 4 / 1920 * log_FK ** 4)
        correction = (1-beta)**2 * alpha**2 / (24 * FK_beta2**2) + rho*beta*nu*alpha / (4 * FK_beta2) + (2 - 3*rho**2)/24 * nu**2

        sigma_B = (alpha / (FK_beta2 * denom)) * (z / x_z) * (1 + correction * T)
        return sigma_B
    
    def implied_vol_smile(self, strikes_s):
        return np.array([self.hagan_implied_vol(K_s) for K_s in strikes_s])

    def implied_vol_smile_mc(self, F_s_T, strikes_s):
        F_bar = F_s_T.mean()
        vols = []
        for K_s in strikes_s:
            if K_s < F_bar:
                price = np.mean(np.maximum(K_s - F_s_T, 0.0))
                v = self.bs_implied_vol(price, F_bar, K_s, self.T, option_type="put")
            else:
                price = np.mean(np.maximum(F_s_T - K_s, 0.0))
                v = self.bs_implied_vol(price, F_bar, K_s, self.T, option_type="call")
            vols.append(v)
        return np.array(vols)

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

    def price(self, K_s, r=0, option_type="call"):
        sigma_B = self.hagan_implied_vol(K_s)
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
    
class EuropeanAnalyticsMRSABR(EuropeanAnalyticsSABR):
    """Mean-reverting SABR analytics"""

    @staticmethod
    def _c_coeff(x, rho):
        if x < 1e-8:
            return 1.0
        ex  = np.exp(-x)
        e2x = np.exp(-2.0 * x)
        term1 = (1.0 - rho**2) * ((1.0 - ex) / x)**2
        term2 = 6.0 * rho**2 * (1.0 - (1.0 + x) * ex) / x**2 * (1.0 - ex) / x
        term3 = 4.0 * rho**2 * ((1.0 - x) * ex - e2x) / x**2
        return term1 + term2 + term3

    @staticmethod
    def mr_effective_params(kappa, T, nu, rho):
        """Effective SABR parameters from the paper
        b_bar is obtained analytically
        c_bar is obtained numerically
        Then: nu_eff = sqrt(c_bar), rho_eff = b_bar / sqrt(c_bar).
        """
        x = kappa * T
        if x < 1e-8:
            return nu, rho

        b_bar = 2.0 * rho * nu / x**2 * (x - 1.0 + np.exp(-x))

        def integrand(t):
            if t < 1e-12:
                return 0.0
            xt = kappa * t
            return t**2 * EuropeanAnalyticsMRSABR._c_coeff(xt, rho)

        c_bar = 3.0 * nu**2 / T**3 * quad(integrand, 0, T, limit=100)[0]

        if c_bar <= 0:
            return 1e-6, 0.0

        nu_eff  = np.sqrt(c_bar)
        rho_eff = float(np.clip(b_bar / nu_eff, -0.999, 0.999))
        return nu_eff, rho_eff

    def __init__(self, F_s, T, beta, nu, rho, sigma_atm, lambda_, kappa):
        self.kappa   = kappa
        self.nu_raw  = nu
        self.rho_raw = rho
        self.nu_eff, self.rho_eff = self.mr_effective_params(kappa, T, nu, rho)
        super().__init__(F_s, T, beta, self.nu_eff, self.rho_eff, sigma_atm, lambda_)