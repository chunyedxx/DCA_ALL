import math
import copy
import numpy as np


class Calomiga():
    def __init__(self, X, Y, lambda_, gamma, alpha, tau, n):
        self.X = X
        self.Y = Y
        self.lambda_ = lambda_
        self.gamma = gamma
        self.alpha = alpha
        self.tau = tau
        self.n = n

    def szt(self, z, t):
        value = 0.0
        if z > 0 and t < abs(z):
            value = z - t
        elif z < 0 and t < abs(z):
            value = z + t
        elif t >= abs(z):
            value = 0
        return value

    def calv(self, omiga):
        v = []
        for ele in omiga:
            if ele >= 0: vi = self.lambda_ * self.gamma * self.alpha * (1 - math.exp(-self.alpha * ele))
            else: vi = - self.lambda_ * self.gamma * self.alpha * (1 - math.exp(self.alpha * ele))
            v.append(vi)
        return np.array(v)

    def func(self, omiga, theta):
        tmp1 = np.linalg.norm(self.Y * theta - self.X * omiga)
        tmp2 = np.linalg.norm(omiga)
        tmp3 = sum(self.alpha * abs(i) for i in omiga)
        G = 1/(2 * self.n) * np.square(tmp1) + self.lambda_ * ((1 - self.gamma)/2 * np.square(tmp2)) + self.gamma * tmp3
        H = self.lambda_ * self.gamma * sum(-1 + self.alpha * abs(i) + math.exp(-self.alpha * abs(i)) for i in omiga)
        F = G - H
        return F

    def funcsos(self, omiga, theta):
        tmp1 = np.linalg.norm(self.Y * theta - self.X * omiga)
        tmp2 = np.linalg.norm(omiga)
        tmp3 = sum(1 - math.exp(-self.alpha * abs(i)) for i in omiga)
        G = 1/(2 * self.n) * np.square(tmp1) + self.lambda_ * ((1 - self.gamma)/2 * np.square(tmp2)) + self.gamma * tmp3
        H = self.lambda_ * self.gamma * sum(-1 + self.alpha * abs(i) + math.exp(-self.alpha * abs(i)) for i in omiga)
        F = G - H
        return F

    def calomiga(self, initomiga, theta):
        # v = self.calv(initomiga)
        while True:
            v = self.calv(initomiga)
            omigaaft = copy.deepcopy(initomiga)
            for j in range(len(initomiga)):
                omigabef_ = np.delete(omigaaft, j, axis=0)
                X_ = np.delete(self.X, j, axis=1)
                tmp = self.X[:, j].T * (self.Y * theta - X_ * omigabef_)
                z = float(1 / self.n * tmp + v[j])
                omigaaft[j] = self.szt(z, self.lambda_ * self.gamma * self.alpha) / (1 + self.lambda_ * (1 - self.gamma))
            if abs(self.func(omigaaft, theta) - self.func(initomiga, theta))[0] <= self.tau * (abs(self.func(initomiga, theta)) + 1)[0]:
                break
            initomiga = copy.deepcopy(omigaaft)
        return omigaaft
