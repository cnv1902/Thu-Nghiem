import statistics
import numpy as np
from numpy.matrixlib import defmatrix
#from src.model import SVC
from sklearn.svm import SVC
import copy

class method:
    @staticmethod
    def own_class_center(X, y):
        X = copy.deepcopy(X)
        pos_index = np.where(y == 1)[0]
        neg_index = np.where(y == -1)[0]
        x_cenpos = np.mean(X[pos_index], axis = 0)
        x_negpos = np.mean(X[neg_index], axis = 0)
        # compute distance 
        X[pos_index] = X[pos_index] - x_cenpos
        X[neg_index] = X[neg_index] - x_negpos
        d_cen = np.linalg.norm(X, axis = 1)
        return d_cen

    @staticmethod
    def own_class_center_divided(X, y):
        X = copy.deepcopy(X)
        X_opposite = copy.deepcopy(X)
        pos_index = np.where(y == 1)[0]
        neg_index = np.where(y == -1)[0]
        x_cenpos = np.mean(X[pos_index], axis = 0)
        x_negpos = np.mean(X[neg_index], axis = 0)
        # compute distance 
        X[pos_index] = X[pos_index] - x_cenpos
        X[neg_index] = X[neg_index] - x_negpos

        X_opposite[pos_index] = X_opposite[pos_index] - x_negpos
        X_opposite[neg_index] = X_opposite[neg_index] - x_cenpos

        d_cen = np.linalg.norm(X, axis = 1)
        d_cen_opposite = np.linalg.norm(X_opposite, axis = 1)
        d_divided = 3*d_cen/d_cen_opposite
        return d_divided
    
    @staticmethod
    def own_class_center_opposite(X, y):
        X1 = copy.deepcopy(X)
        pos_index = np.where(y == 1)[0]
        neg_index = np.where(y == -1)[0]
        x_cenpos = np.mean(X1[pos_index], axis = 0)
        x_negpos = np.mean(X1[neg_index], axis = 0)
        # compute distance 
        X1[pos_index] = X1[pos_index] - x_negpos
        X1[neg_index] = X1[neg_index] - x_cenpos 
        d_cen = 1/(np.linalg.norm(X1, axis = 1))
        return d_cen

    @staticmethod
    def distance_center_own_opposite_tam(X,y):
        X_own = copy.deepcopy(X)
        X_opposite = copy.deepcopy(X)
        pos_index = np.where(y == 1)[0]
        neg_index = np.where(y == -1)[0]
        x_cenpos = np.mean(X[pos_index], axis=0)
        x_cenneg = np.mean(X[neg_index], axis=0)

        # compute distance
        X_own[pos_index] = X_own[pos_index] - x_cenpos
        X_own[neg_index] = X_own[neg_index] - x_cenneg

        X_opposite[pos_index] = X_opposite[pos_index] - x_cenneg
        X_opposite[neg_index] = X_opposite[neg_index] - x_cenpos

        x_2tam = x_cenneg - x_cenpos

        d_cen_own = np.linalg.norm(X_own, axis=1)
        d_cen_opposite = np.linalg.norm(X_opposite, axis=1)
        d_tam = np.linalg.norm(x_2tam, axis=0)
        return d_cen_own, d_cen_opposite, d_tam

    @staticmethod
    def estimated_hyper_lin(X, y):
        # compute center
        X1 = copy.deepcopy(X)
        x_cen = np.mean(X1, axis = 0)
        # compute distance 
        X1 = X1 - x_cen
        d_cen = np.linalg.norm(X1, axis = 1)   
        return d_cen

    @staticmethod
    def actual_hyper_lin(X, y, kernel = 'rbf', C = None, gamma = None):
        # compute support vertor
        cls = SVC()
        cls.fit(X, y)
        d = y * cls.decision_function(X)
        return d

class function:
    @staticmethod
    def lin(d, delta=1e-6):
        dmax = np.max(d)
        return 1 - d / (dmax + delta)

    @staticmethod
    def lin_center_own(d, pos_ind,neg_ind,delta=1e-6):
        f = np.zeros(len(d))
        dmax_pos = np.max(d[pos_ind])
        dmax_neg = np.max(d[neg_ind])
        f[pos_ind] = 1 - d[pos_ind] / (dmax_pos+delta)
        f[neg_ind] = 1 - d[neg_ind] / (dmax_neg+delta)
        return f

    @staticmethod
    def exp(d, beta):
        return 2 / (1 + np.exp(beta * d))
    @staticmethod
    def gau(d, u, sigma):
        return np.exp(-np.linalg.norm(d-u)**2/(2 * sigma**2))

    @staticmethod
    def func_own_opp (d_cenpos, d_cenneg, pos_ind, neg_ind,d_tam):
        f = np.zeros(len(d_cenpos))
        f[pos_ind] = (d_cenneg[pos_ind] + d_cenpos[pos_ind]) / (d_cenpos[pos_ind] + 2*d_tam)
        f[neg_ind] = (d_cenpos[neg_ind] + d_cenneg[neg_ind]) / (d_cenneg[neg_ind] + 2*d_tam)
        return f

    @staticmethod
    def func_own_opp_new(d_cenpos, d_cenneg, pos_ind, neg_ind, d_tam, delta=1e-6):
        f = np.zeros(len(d_cenpos))
        f[pos_ind] = d_cenneg[pos_ind]/(d_cenpos[pos_ind]+d_tam+delta)
        f[neg_ind] = d_cenpos[neg_ind]/(d_cenneg[neg_ind]+d_tam+delta)
        return f

    @staticmethod
    def func_own_opp_new_v1(d_cenpos, d_cenneg, pos_ind, neg_ind, d_tam, delta=1e-6):
        f = np.zeros(len(d_cenpos))
        dmax_pos = np.max(d_cenpos[pos_ind])
        dmin_pos = np.min(d_cenpos[neg_ind])
        dmax_neg = np.max(d_cenneg[neg_ind])
        dmin_neg = np.min(d_cenneg[pos_ind])
        f[pos_ind] = 1 - (d_cenpos[pos_ind] + dmin_neg/d_cenneg[pos_ind])/(dmax_pos+dmin_neg+d_tam+delta)
        f[neg_ind] = 1 - (d_cenneg[neg_ind] + dmin_pos/d_cenpos[neg_ind])/(dmax_neg+dmin_pos+d_tam+delta)
        return f

    @staticmethod
    def func_own_opp_new_v2(d_cenpos, d_cenneg, pos_ind, neg_ind, d_tam, delta=1e-6):
        f = np.zeros(len(d_cenpos))
        dmax_pos = np.max(d_cenpos[pos_ind])
        dmin_pos = np.min(d_cenpos[neg_ind])
        dmax_neg = np.max(d_cenneg[neg_ind])
        dmin_neg = np.min(d_cenneg[pos_ind])
        f[pos_ind] = 1 - (d_cenpos[pos_ind] + d_cenneg[pos_ind])/(dmax_pos+dmin_neg+d_tam+delta)
        f[neg_ind] = 1 - (d_cenneg[neg_ind] + d_cenpos[neg_ind])/(dmax_neg+dmin_pos+d_tam+delta)
        return f