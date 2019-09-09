import math
import copy
import numpy as np


class Calomiga():
    def __init__(self, X, Y, lambda1, lambda2, alpha, tau, n):
        self.X = X
        self.Y = Y
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.alpha = alpha
        self.tau = tau
        self.n = n

# 算法2.1求V
    def calv(self, omiga):
        v = []
        for ele in omiga:
            if self.alpha * (ele ** 2) >= 1: 
                vi = 2 * self.alpha * ele
                vi = vi[0,0]
            else: vi = 0    
            v.append(vi)    
        return np.mat(v).reshape([-1,1])

    def funcsos(self, omiga, theta):
        tmp1 = np.square(np.linalg.norm(self.Y * theta - self.X * omiga))
        return tmp1 + self.lambda1 * np.square(np.linalg.norm(omiga)) + self.lambda2 * np.linalg.norm(omiga, ord=1)

# 函数H-mui-omiga
    def funcH(self, omiga, theta, mui):
        rows, cols = np.shape(self.X)
        eye = np.mat(np.identity(cols))
        tmp = sum(-1 + max(self.alpha * i ** 2, 1) for i in omiga)
        H = omiga.T * (mui * eye - self.X.T * self.X) * omiga + 2 * omiga.T * self.X.T * self.Y * theta + self.lambda2 * tmp
        return H

    def func(self, omiga, theta, mui):
        G = omiga.T * (mui + self.lambda1 + self.alpha * self.lambda2) * omiga
        H = self.funcH(omiga, theta, mui)
        F = G - H
        return F

    def calomiga(self, initomiga, theta):
        while True:
            mui = np.square(np.linalg.norm(self.X, ord="fro"))
            v = 2  * (mui * np.mat(np.identity(4)) - self.X.T * self.X) * initomiga + 2 * self.X.T * self.Y * theta + self.lambda2 * self.calv(initomiga)
            omigaaft = 1 / (2 * (mui + self.lambda1 + self.alpha * self.lambda2)) * v
            while self.funcH(omigaaft, theta, mui) > self.funcH(initomiga, theta, mui) + abs((omigaaft - initomiga).T * v):
                initomiga = copy.deepcopy(omigaaft)
                mui = self.tau * mui
                v = 2  * (mui * np.eye(4) - self.X.T * self.X) * initomiga + 2 * self.X.T * self.Y * theta + self.lambda2 * self.calv(initomiga)
                omigaaft = 1 / (2 * (mui + self.lambda1 + self.alpha * self.lambda2)) * v
            if abs(self.func(omigaaft, theta, mui) - self.func(initomiga, theta, mui))[0] <= self.tau * (abs(self.func(initomiga, theta, mui)) + 1)[0]:
                break
            initomiga = copy.deepcopy(omigaaft)
        return omigaaft