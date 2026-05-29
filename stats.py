import numpy as np
import scipy as sc

class PathStatistics:
    @staticmethod
    def mean(paths):
        return np.mean(paths, axis=0)
    @staticmethod
    def variance(paths):
        return np.var(paths, axis=0)
    @staticmethod
    def skewness(paths):
        return sc.stats.skew(paths, axis=0)
    @staticmethod
    def kurtosis(paths):
        return sc.stats.kurtosis(paths, axis=0)
    @staticmethod
    def quantile(paths, q):
        return np.percentile(paths, q, axis=0)
    @staticmethod
    def max(paths):
        return np.max(paths, axis=0)
    @staticmethod
    def min(paths):
        return np.min(paths, axis=0)
    @staticmethod
    def std_dev(paths):
        return np.std(paths, axis=0)