# svm implementation
"""
this implementation is based on the paper:
Large-scale Multiclass Support Vector Machine Training via Euclidean Projection onto the Simplex.
Mathieu Blondel, Akinori Fujino, and Naonori Ueda.
ICPR 2014.
"""

import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_random_state
from sklearn.preprocessing import LabelEncoder


class MulticlassSVM(BaseEstimator, ClassifierMixin):

    def __init__(self, C=1, max_iter=50, tol=0.05,
                 random_state=None, verbose=0):
        self.C = C
        self.max_iter = max_iter
        self.tol = tol,
        self.random_state = random_state
        self.verbose = verbose # used to control the message outputing

    def projection_simplex(self, v, z=1):
        """
        Projection onto the simplex:
            w^* = argmin_w 0.5 ||w-v||^2 s.t. \sum_i w_i = z, w_i >= 0
        """
        # For other algorithms computing the same projection, see
        # https://gist.github.com/mblondel/6f3b7aaad90606b98f71
        numFeatures = v.shape[0]
        u = np.sort(v)[::-1]
        cssv = np.cumsum(u) - z
        ind = np.arange(numFeatures) + 1
        cond = u - cssv / ind > 0
        rho = ind[cond][-1]
        theta = cssv[cond][-1] / float(rho)
        w = np.maximum(v - theta, 0)
        return w

    def _partial_gradient(self, X, y, i):
        # Partial gradient for the ith sample.
        g = np.dot(X[i], self.W.T) + 1
        g[y[i]] -= 1
        return g

    def _violation(self, g, y, i):
        # Optimality violation for the ith sample.
        smallest = np.inf
        for k in xrange(g.shape[0]):
            if k == y[i] and self.dual_W[k, i] >= self.C:
                continue
            elif k != y[i] and self.dual_W[k, i] >= 0:
                continue

            smallest = min(smallest, g[k])

        return g.max() - smallest

    def _solve_subproblem(self, g, y, norms, i):
        # Prepare inputs to the projection.
        Ci = np.zeros(g.shape[0])
        Ci[y[i]] = self.C
        beta_hat = norms[i] * (Ci - self.dual_W[:, i]) + g / norms[i]
        z = self.C * norms[i]

        # Compute projection onto the simplex.
        beta = self.projection_simplex(beta_hat, z)

        return Ci - self.dual_W[:, i] - beta / norms[i]

    def fit(self, X, y):
        print 'start fiting ..'
        numSamples, numFeatures = X.shape

        # X, list of samples
        # sample, list of features? value or key, I think it list of feature values
        
        
        # numSamples, numSamples
        # numFeatures, numFeatures
        # numClasses, numLabels
        
        # Normalize labels.
        self.labelEncoder = LabelEncoder() # labelEncoder, based on dict or hash
        # labelEncoder, first fit and then can be transform back to original label
        '''
        fit(y)  Fit label encoder
        fit_transform(y)    Fit label encoder and return encoded labels
        get_params([deep])  Get parameters for this estimator.
        inverse_transform(y)    Transform labels back to original encoding.
        set_params(**params)    Set the parameters of this estimator.
        transform(y)    Transform labels to normalized encoding.
        '''
        y = self.labelEncoder.fit_transform(y)

        # Initialize primal and dual coefficients.
        numClasses = len(self.labelEncoder.classes_) # number of labels
        self.dual_W = np.zeros((numClasses, numSamples), dtype=np.float64)
        # self.dual_coef used to solve subproblem
        # m * n matrix, m = numLabels, n = numSamples
        self.W = np.zeros((numClasses, numFeatures))
        # m * n matrix, m = numLabels, n = numFeatures


        
        # Pre-compute norms.
        norms = np.sqrt(np.sum(X ** 2, axis=1))

        # Shuffle sample indices.
        rs = check_random_state(self.random_state)
        ind = np.arange(numSamples)
        '''
        >>> np.arange(3)
        array([0, 1, 2])
        '''
        rs.shuffle(ind)

        violation_init = None
        for it in xrange(self.max_iter):
            violation_sum = 0

            for ii in xrange(numSamples):
                i = ind[ii] # randomly choose a sample
                
                # All-zero samples can be safely ignored.
                if norms[i] == 0:
                    continue

                g = self._partial_gradient(X, y, i)
                v = self._violation(g, y, i)
                violation_sum += v

                if v < 1e-12:
                    continue

                # Solve subproblem for the ith sample.
                delta = self._solve_subproblem(g, y, norms, i)

                # Update primal and dual coefficients.
                self.W += (delta * X[i][:, np.newaxis]).T #transpose newaxis:none
                self.dual_W[:, i] += delta

            if it == 0:
                violation_init = violation_sum

            vratio = violation_sum / violation_init

            if self.verbose >= 1:
                print "iter", it + 1, "violation", vratio

            if vratio < self.tol:
                if self.verbose >= 1:
                    print "Converged"
                break

        return self

    def predict(self, X):
        decision = np.dot(X, self.W.T)
        pred = decision.argmax(axis=1)
        return self.labelEncoder.inverse_transform(pred)


if __name__ == '__main__':
    from sklearn.datasets import load_iris

    iris = load_iris()
    X, y = iris.data, iris.target
    print 'type of X: %s' % type(X)
    print 'type of y: %s' % type(y)
    #print X
    #print X[0]
    #print X[0][0]
    #print y
    clf = MulticlassSVM(C=0.1, tol=0.01, max_iter=100, random_state=0, verbose=0)
    clf.fit(X, y)
    #print clf.score(X, y)