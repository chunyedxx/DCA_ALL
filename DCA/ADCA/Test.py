import numpy as np
import numpy.linalg as nl
from DataLoader import getData, getLabel2id, getLabelid


def test(path, omigaset, mui):
    X, labels = getData(path)
    labels2id, labnum = getLabel2id(labels)
    labelsid = getLabelid(labels, labels2id)
    rows, cols = np.shape(X)
    corre = 0
    for i in range(rows):
        predicts = []
        for ind in range(len(labels2id)):
                pred = np.square(nl.norm(np.matrix(X[i, :]) * omigaset - mui[ind] * omigaset))
                predicts.append(pred)
        if labelsid[i] == predicts.index(min(predicts)):
                corre += 1
    pre = corre / len(labels)
    return pre