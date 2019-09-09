import numpy as np
from Test import test
import numpy.linalg as nl
# from DCA1 import Calomiga
from DCA2 import Calomiga
from CalTheta import Caltheta
from DataLoader import getData, getMui, getLabel2id, getLabelid, getLabMatrix


def initial(X, Y, D, Q, Q_M, p):
    np.random.seed(123)
    eye = np.mat(np.identity(Q))
    omiga = np.mat(np.random.rand(p, 1))
    s = (eye - Q_M * Q_M.T * D) * D.I * Y.T * X * omiga
    theta = s / np.sqrt(s.T * D * s)
    return omiga, theta


def haha(nparray):
    [rows, cols] = np.shape(nparray)
    for i in range(rows):
        for j in range(cols):
            if nparray[i, j] <= 0.0:
                nparray[i, j] = 0
    return nparray


def main(lambda1, lambda2, alpha, tau):
    X, labels = getData('Dataset\\iris\\iris.data')  # 类别数 3
    Y, Q = getLabMatrix(labels)
    muii = getMui(X, labels)
    p = np.shape(X)[1]   # 特征向量维数 3
    n = np.shape(X)[0]   # 数据点个数 150
    D = 1 / n * Y.T * Y   # Q * Q 维 3 * 3
    calomiga = Calomiga(X, Y, lambda1, lambda2, alpha, tau, n)
    caltheta = Caltheta(X, Y)
    Q_M = np.mat(np.ones(shape=[Q, 1]))
    omigaset = []
    for epo in range(0, 5):
        initomiga, inittheta = initial(X, Y, D, Q, Q_M, p)
        initmui = np.linalg.norm(X, ord="fro")
        while True:
            omigaaft = calomiga.calomiga(initomiga, inittheta)
            print(omigaaft)
            thetaaft = caltheta.caltheta(omigaaft, Q_M)
            if abs(calomiga.funcsos(omigaaft, thetaaft) - calomiga.funcsos(initomiga, inittheta)) <= 1e-2:
                break
            inittheta = thetaaft
            initomiga = omigaaft
        Q_M = np.hstack((Q_M, thetaaft))
        if omigaset == []:
            omigaset = omigaaft
        else:
            omigaset = np.hstack((omigaset, omigaaft))
    omigaset = np.mat(omigaset)
    print(omigaset)
    pre = test('Dataset\\iris\\test.data', omigaset, muii)
    print(pre)


if __name__ == "__main__":
    try:
        alpha = 25
        lambda1 = 0.01  # 0.01 0.02,0.03, 0.04 0.05 0.06 0.08 0.012 0.1 0.15 0.4 0.6 0.7 0.9
        lambda2 = 0.5  # 0.1-0.9
        tau = 1e-5  # 0.5左右
        main(lambda1, lambda2, alpha, tau)
    except KeyboardInterrupt as e:
        print("[STOP]", e)
