import numpy as np
import numpy.linalg as lin
from  numpy.random import randn
import scipy as sp
import scipy.stats
import math
import matplotlib.pyplot as plt

class SyntheticData:
    """Initialization"""
    def __init__(self, n_samples = 200):
        self._n_samples = n_samples
        # mean vectors
        self._mu0 = np.array([0, 0], dtype=float)
        self._mu1 = np.array([1.5, -1.0], dtype=float)
        self._mu2 = np.array([1.1, 1.6], dtype=float)
        # covariance symmetric matrices
        self._sigma0 = np.array([1.0, -0.2, -0.2, 0.9], dtype =float).reshape((2,2))
        self._sigma1 = np.array([0.6, 0.6, 0.6, 1.2], dtype=float).reshape((2,2))
        self._sigma2 = np.array([0.5, -0.6, -0.6, 1.0], dtype=float).reshape((2,2))
        # precision matrices
        self._precision0 = lin.inv(self._sigma0)
        self._precision1 = lin.inv(self._sigma1)
        self._precision2 = lin.inv(self._sigma2)
        # determinant of precision
        self._det_prec0 = lin.det(self._precision0)
        self._det_prec1 = lin.det(self._precision1)
        self._det_prec2 = lin.det(self._precision2)
        # mix coefficients and prior distrubutions
        self._prior = 0.5
        self._etha = 0.6
        self._pi1 = self._etha
        self._pi2 = 1 - self._etha
    """Obtain data points from each class"""
    def gen_data(self):
        # prior distribution
        prior_dist = sp.stats.binom(self._n_samples, self._prior)
        n0_samples = prior_dist.rvs(1)[0]
        n12_samples = self._n_samples - n0_samples

        pi_dist = sp.stats.binom(n12_samples, self._etha)
        n1_samples = pi_dist.rvs(1)[0]
        n2_samples = n12_samples - n1_samples

        # target variable and corresponding colors
        target0 = np.array([0] * n0_samples).T
        color0 = np.array(["RED"] * n0_samples).T

        target1 = np.array([1] * n12_samples).T
        color1 = np.array(["BLUE"] * n12_samples).T

        color = np.r_[color0, color1]
        target = np.r_[target0, target1]

        # likelihood
        X = np.r_[np.dot(randn(n0_samples, 2), self._sigma0) + self._mu0,
                np.dot(randn(n1_samples, 2), self._sigma1) + self._mu1,
                np.dot(randn(n2_samples, 2), self._sigma2) + self._mu2]
        return X, target, color, self._n_samples

    """Modify sample number"""
    def set_samples(self, n):
        self._n_samples = n
        return

    """Function to draw decision boundary"""
    def _delta(self, x, y, mu, precision):
        mux = mu[0]
        muy = mu[1]
        lambdaxx = precision[0][0]
        lambdaxy = precision[0][1]
        lambdayx = precision[1][0]
        lambdayy = precision[1][1]
        term1 = (x - mux) * lambdaxx * (x - mux)
        term2 = (x - mux) * lambdaxy * (y - muy)
        term3 = (y - muy) * lambdayx * (x - mux)
        term4 = (y - muy) * lambdayy * (y - muy)
        return (term1 + term2 + term3 + term4)/(-2)

    def _D2Gaussian(self, x, y, mu, precision, det_prec):
        return (det_prec)*np.exp(self._delta(x, y, mu, precision))/(2 * math.pi)

    def f(self, x1, x2):
        term1 = self._pi1 * self._D2Gaussian(x1, x2, self._mu1, self._precision1, self._det_prec1)
        term2 = self._pi2 * self._D2Gaussian(x1, x2, self._mu2, self._precision2, self._det_prec2)
        term3 = self._D2Gaussian(x1, x2, self._mu0, self._precision0, self._det_prec0)
        result = term1 + term2 - term3
        return result

"""Test code"""
if __name__ == '__main__':
    distribution = SyntheticData()
    distribution.set_samples(30)
    X, T, color, n_samples = distribution.gen_data()
    plt.scatter(X[:,0], X[:,1], c = color)
    plt.show()
