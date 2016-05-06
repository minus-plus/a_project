# svm implementation
"""
this implementation is based on the paper:
Large-scale Multiclass Support Vector Machine Training via Euclidean Projection onto the Simplex.
Mathieu Blondel, Akinori Fujino, and Naonori Ueda.
ICPR 2014.
"""
# polynomial kernel works better, gaussian kernel need to tune
#
#
#

import sys
import time

import numpy as np
from numpy import linalg

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_random_state
from sklearn.preprocessing import LabelEncoder

# define kernels 
def linear_kernel(x1, x2):
    return np.dot(x1, x2)

sigma_list = [0.1, 0.5, 1, 1.5, 2, 3, 4, 5]
def gaussian_kernel(x, y, sigma=10, gamma=0.05):
    #return np.exp(-linalg.norm(x-y)**2 / (2 * (sigma ** 2)))
    return np.exp(gamma * (-linalg.norm(x - y) ** 2))
def polynomial_kernel(x, y, p=1.5):
    return (1 + np.dot(x, y)) ** p

class MulticlassSVM(BaseEstimator, ClassifierMixin):
    def __init__(self, C=1, max_iteration=50, tolorance=0.0005,
                 random_state=None, verbose=0):
        max_iteration = 200
        verbose = 1
        C = 1
        self.C = C
        self.max_iteration = max_iteration
        self.tolorance = 1e-8
        self.random_state = random_state
        self.verbose = verbose # used to control the message outputing
        self.kernel = gaussian_kernel

    def get_kernel_matrix(self, X1, X2):
        K = np.zeros(((len(X1), len(X2))))
        for i in range(len(X1)):
            for j in range(len(X2)):
                K[i][j] = self.kernel(X1[i], X2[j])
        return K

    def projection_simplex(self, v, z=1):

        numFeatures = v.shape[0]
        u = np.sort(v)[::-1]
        cssv = np.cumsum(u) - z
        index = np.arange(numFeatures) + 1
        cond = u - cssv / index > 0
        rho = index[cond][-1]
        theta = cssv[cond][-1] / float(rho)
        w = np.maximum(v - theta, 0)
        return w
    # equation 4
    def get_partial_gradient(self, X, y, i):
        # Partial gradient for the ith sample.
        g = np.dot(self.alpha, self.K[:, i]) + 1
        g[y[i]] -= 1
        return g

    # equation 5
    def get_violation(self, g, y, i):
        # Optimality violation for the ith sample.
        smallest = np.inf
        for k in xrange(g.shape[0]):
            if k == y[i] and self.alpha[k, i] >= self.C:
                continue
            elif k != y[i] and self.alpha[k, i] >= 0:
                continue
            smallest = min(smallest, g[k])

        return g.max() - smallest
    # equation 6, get delta
    def solve_subproblem(self, g, y, norms, i):
        # Prepare inputs to the projection.
        Ci = np.zeros(g.shape[0])
        Ci[y[i]] = self.C
        beta_hat = norms[i] * (Ci - self.alpha[:, i]) + g / norms[i]
        z = self.C * norms[i]
        # Compute projection onto the simplex.
        beta = self.projection_simplex(beta_hat, z)
        return Ci - self.alpha[:, i] - beta / norms[i]

    def fit(self, X, y):
        numSamples, numFeatures = X.shape
        self.X = X
        #K = np.zeros(numSamples, numSamples)
        self.labelEncoder = LabelEncoder() 
        y = self.labelEncoder.fit_transform(y)
        # Initialize primal and dual coefficients.
        numClasses = len(self.labelEncoder.classes_)
        self.alpha = np.zeros((numClasses, numSamples), dtype=np.float64)
        self.W = np.zeros((numClasses, numFeatures))
        # pre-compute kernel matrix
        norms = np.zeros((numSamples))
        # K-kernel matrix
        K = np.zeros((numSamples, numSamples))
        for i in range(numSamples):
            for j in range(numSamples):
                K[i][j] = self.kernel(X[i], X[j])
        # Pre-compute norms. what is norms for
        self.K = self.get_kernel_matrix(X, X)
        for s in range(numSamples):
            norms[s] = np.sqrt(K[i][i])
        randomState = check_random_state(self.random_state)
        index = np.arange(numSamples)
        randomState.shuffle(index)

        violation_init = None
        for it in xrange(self.max_iteration):
            violation_sum = 0
            for ind in xrange(numSamples):
                i = index[ind]
                # ignore zero sample
                if norms[i] == 0:
                    continue
                # compute g_i by equation 4
                g = self.get_partial_gradient(X, y, i)
                # compute v_i by equation 5
                v = self.get_violation(g, y, i)
                violation_sum += v
                if v < 1e-12:
                    continue
                # Solve subproblem for the ith sample.
                # compute delta_i by equation 6
                delta = self.solve_subproblem(g, y, norms, i)
                self.alpha[:, i] += delta    
            if it == 0:
                violation_init = violation_sum
            vratio = violation_sum / violation_init
            if self.verbose >= 1:
                print "iter", it + 1, "violation", vratio
            if vratio < self.tolorance:
                if self.verbose >= 1:
                    print "Converged"
                break
        self.W = np.dot(self.alpha, X) #transpose newaxis:none
        return self

    def predict(self, X):
        K = self.get_kernel_matrix(X, self.X) # 50 * 451
        decision = np.dot(K, self.alpha.T) # 50 * 451 dot 2 * 451.T --> 50 * 2
        pred = decision.argmax(axis=1)
        return self.labelEncoder.inverse_transform(pred)

