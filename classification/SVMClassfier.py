#define kernels
import numpy as np
from numpy import linalg

def linear_kernel(x1, x2):
    return np.dot(x1, x2)
def gaussian_kernel(x, y, sigma=5.0):
    return np.exp(-linalg.norm(x-y)**2 / (2 * (sigma ** 2)))

class SVM(object):
    def __init__(self, kernel=gaussian_kernel, C=None): # C?
        self.kernel = kernel
        self.C = C
        if not self.C:
            self.C = float(self.c)
    def fit(self, X, y):
        # train method will import trainingData, a list of datum(util.Counter)
        # here, X is trainingData
        # y is label?
        
        numSample = len(X)
        features = X.keys()
        numFeatures = len(features)
        # Gram matrix
        K = np.zeros(numSample, numSample) # number of n * n kernels
        
        for i in range(numSample):
            for j in range(numSample):
                K[i][j] = self.kernel(X[i], X[j]) # sample i and sample j
        
        # compute p
        P = cvxopt.matrix(np.outer(y,y) * K)
        q = cvxopt.matrix(np.ones(numSample) * -1)
        A = cvxopt.matrix(y, (1,numSample))
        b = cvxopt.matrix(0.0)
        
        if not self.C:
            G = cvxopt.matrix(np.diag(np.ones(numSample) * -1))
            h = cvxopt.matrix(np.zeros(numSample))
        else:
            tmp1 = np.diag(np.ones(numSample) * -1)
            tmp2 = np.identity(numSample)
            G = cvxopt.matrix(np.vstack((tmp1, tmp2)))
            tmp1 = np.zeros(numSample)
            tmp2 = np.ones(numSample) * self.C
            h = cvxopt.matrix(np.hstack((tmp1, tmp2)))            
            
        # then solve the qp problem
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)
        
        # Lagrange multipliers
        a = np.ravel(solution['x'])   # ai s
        
        # get Support vectors have non zero lagrange multipliers
        sv = a > 1e-5
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            