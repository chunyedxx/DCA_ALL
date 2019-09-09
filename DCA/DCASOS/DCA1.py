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

    def func(self, omiga, theta, v):
        F = omiga.T * (self.X.T * self.X + (self.lambda1 + self.lambda2 * self.alpha) * np.mat(np.identity(4))) * omiga - omiga.T * (2 * self.X.T * self.Y * theta + self.lambda2 * v)
        return F

    def funcsos(self, omiga, theta):
        tmp1 = np.square(np.linalg.norm(self.Y * theta - self.X * omiga))
        return tmp1 + self.lambda1 * np.square(np.linalg.norm(omiga)) + self.lambda2 * np.linalg.norm(omiga, ord=1)

    def calomiga(self, initomiga, theta):
        while True:
            v = self.calv(initomiga)
            M = np.mat((self.lambda1 + self.alpha * self.lambda2) * np.mat(np.identity(4)))
            omigaaft = 1 / 2 * (M.I - M.I * self.X.T * (np.mat(np.identity(120)) + self.X * M.I * self.X.T).I * self.X * M.I) * (2 * self.X.T * self.Y * theta + self.lambda2 * v)
            if abs(self.func(omigaaft, theta, v) - self.func(initomiga, theta, v))[0] <= self.tau * (abs(self.func(initomiga, theta, v)) + 1)[0]:
                break
            initomiga = copy.deepcopy(omigaaft)
        return omigaaft