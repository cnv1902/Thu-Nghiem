import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split as tts
import pandas as pd
from . import methods
class Wsvm():
    def __init__(self,C,distribution_weight):
        self.C = C
        self.w = None
        self.b = None
        self.distribution_weight = distribution_weight

    def fit(self, X, y):
        # preprocess:
        X = np.array(X)
        y = np.array(y)
        N, d = X.shape
        # print(type(self.distribution_weight))
        P, q, G, h, A, b = methods.dual_problem_quadratic_program(X, y, self.C, self.distribution_weight)
        #Solve Quadratic Program
        sol = methods.dual_problem_quadratic_solver(P, q,G, h, A, b)

        # Caculate Lagrange 
        lam = methods.svm_lagrange_mutipliers(sol)

        # Find Svm suport vectors that lam > 0
        S = methods.svm_support_vectors(lam)

        # Find weight
        self.w = methods.svm_weight(X, y, lam)
        # Find bias 
        self.b = methods.svm_bias(X, y, S, self.w)
        self.w = np.array(self.w)
        self.b = np.array(self.b)

    def predict(self, X):
        X2 = np.array(X)
        H = np.sign(X2.dot(self.w)+self.b)
        return H