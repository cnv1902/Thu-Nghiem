from .svm import solver
from .svm.kernel import kernel
import numpy as np
class SVC:
    def __init__(self,X = None,y = None, kernel_name = 'linear', C= None, gamma = None,r = None, d = None, distribution_weight = None):
        self.kernel_name = kernel_name
        self.C = C
        self.gamma = gamma
        self.r = r
        self.d = d
        self.distribution_weight = distribution_weight
        #self.kernel = kernel(self.kernel_name, self.gamma, self.r, self.d)
        # self.X = X
        # self.y = y
    def fit(self, X, y):
        self.lam = solver.fit(X, y, C = self.C, distribution_weight= self.distribution_weight,
                                 kernel_name= self.kernel_name, gamma = self.gamma, r = self.r, d = self. d)
        # Find support vertor index
        self.S = self.find_support_vectors(self.lam)
        # Find support vertor on margin index
        self.M = self.find_margin_vertors(self.lam, self.C)
        # Find weight
        if self.kernel_name == 'linear':
            self.w = self.compute_weight(X, y)
        self.b = self.compute_bias(X, y)
        self.X = X  
        self.y = y
    
    def compute_bias(self, X, y):
        if self.kernel_name == 'linear':
            return np.mean(y[self.S] - np.dot(X[self.S],self.w))
        else:
            num_M =  self.M.shape[0]
            y_hat = np.zeros(self.M.shape[0], dtype= np.float32)
            # dataset is a support vector
            X_sv = X[self.S]
            y_sv = y[self.S]
            # dataset is on the margin
            X_margin = X[self.M]
            y_margin = y[self.M]
            for i in range (num_M):
                y_hat[i]  = y_margin[i] - np.sum(self.lam[self.S] * y_sv * self.kernel.compute(X_sv, X_margin[i])) 
            return np.sum(y_hat) / num_M

    def compute_weight(self, X, y):
        return np.dot(X.T,(y[:,np.newaxis]*self.lam)).flatten()


    def decision_function(self, X):
        if self.kernel_name == 'linear':
            return X.dot(self.w) + self.b
        else:
            N = X.shape[0]
            y_predict = np.zeros(N)
            # dataset is a support vector
            X_sv = self.X[self.S]
            y_sv = self.y[self.S]
            # compute 
            for i in range (N):
                y_predict[i] = np.sum(self.lam[self.S] * y_sv * self.kernel.compute(X_sv, X[i]))
            return y_predict + self.b
    
    def predict(self, X):
        return np.sign(self.decision_function(X))
    @staticmethod
    def find_support_vectors(lam):
        return np.where(lam >= 1e-2)[0]
    @staticmethod
    def find_margin_vertors(lam, C):
        return np.where(np.logical_and((lam > 0), (lam < C)))[0]