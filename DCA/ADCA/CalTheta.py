import numpy as np


class Caltheta():
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def caltheta(self, omiga, Q):
        D = 1 / len(self.X) * self.Y.T * self.Y
        aa = np.identity(3)
        s = (aa - Q * Q.T * D) * D.I * self.Y.T * self.X * omiga
        theta1 = s / np.sqrt(s.T * D * s)
        return theta1
