# svm implementation
"""
this implementation is based on the paper:
Large-scale Multiclass Support Vector Machine Training via Euclidean Projection onto the Simplex.
Mathieu Blondel, Akinori Fujino, and Naonori Ueda.
ICPR 2014.
"""
import sys

import numpy as np
from numpy import linalg

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_random_state
from sklearn.preprocessing import LabelEncoder

# define kernels
def linear_kernel(x1, x2):
    return np.dot(x1, x2)
def gaussian_kernel(x, y, sigma=5.0):
    return np.exp(-linalg.norm(x-y)**2 / (2 * (sigma ** 2)))


class MulticlassSVM(BaseEstimator, ClassifierMixin):

    def __init__(self, C=1, max_iteration=50, tolorance=0.05,
                 random_state=None, verbose=0):
        self.C = C
        self.max_iteration = max_iteration
        self.tolorance = tolorance
        self.random_state = random_state
        self.verbose = verbose # used to control the message outputing
        self.kernel = linear_kernel

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

    def get_partial_gradient(self, X, y, i):
        # Partial gradient for the ith sample.
        g = np.dot(X[i], self.W.T) + 1
        g[y[i]] -= 1
        return g

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

        # X, list of samples

        # Normalize labels.
        #K = np.zeros(numSamples, numSamples)
        self.labelEncoder = LabelEncoder() 
        y = self.labelEncoder.fit_transform(y)

        # Initialize primal and dual coefficients.
        numClasses = len(self.labelEncoder.classes_)
        self.alpha = np.zeros((numClasses, numSamples), dtype=np.float64)
        self.W = np.zeros((numClasses, numFeatures))
        # Pre-compute norms. what is norms for
        norms = np.sqrt(np.sum(X ** 2, axis=1))

        # Shuffle sample indexices.
        randomState = check_random_state(self.random_state)
        index = np.arange(numSamples)
        randomState.shuffle(index)

        violation_init = None
        for it in xrange(self.max_iteration):
            violation_sum = 0
            for ind in xrange(numSamples):
                i = index[ind] # randomly choose a sample
                
                # ignore zero sample
                if norms[i] == 0:
                    continue
                g = self.get_partial_gradient(X, y, i)
                v = self.get_violation(g, y, i)
                violation_sum += v
                if v < 1e-12:
                    continue
                # Solve subproblem for the ith sample.
                # here is delta_i
                delta = self.solve_subproblem(g, y, norms, i)
                self.alpha[:, i] += delta    

                # Update primal and dual coefficients.
                # w(alpha_i) = delta_i * x_i
                self.W += (self.alpha[:, i] * X[i][:, np.newaxis]).T #transpose newaxis:none
                #slef.W = 
                # alpha_i = delta_i + delta_i
            if it == 0:
                violation_init = violation_sum
            vratio = violation_sum / violation_init

            if self.verbose >= 1:
                print "iter", it + 1, "violation", vratio

            if vratio < self.tolorance:
                if self.verbose >= 1:
                    print "Converged"
                break

        return self

    def predict(self, X):
        decision = np.dot(X, self.W.T)
        pred = decision.argmax(axis=1)
        return self.labelEncoder.inverse_transform(pred)

