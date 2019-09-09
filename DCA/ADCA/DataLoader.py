import numpy as np
from sklearn import preprocessing


def getData(path):
    inputexamples = readData(path)
    features = []
    labels = []
    for example in inputexamples:
        fea, lab = example[0], example[1]
        features.append(fea)
        labels.append(lab)
    features = np.matrix(preprocessing.scale(np.reshape(np.array(features), newshape=[len(labels), -1])))
    return features, labels


def getLabel2id(labels):
    labels2id = {}
    labnum = 0
    for lab in labels:
        if lab not in labels2id.keys():
                labels2id[lab] = labnum
                labnum += 1
    return labels2id, labnum


def getLabMatrix(labels):
    labels2id, labnum = getLabel2id(labels)
    labelsid = getLabelid(labels, labels2id)
    Y = np.zeros(shape=[len(labels), labnum])
    for num, id in enumerate(labelsid): Y[num][id] = 1
    return np.matrix(Y), labnum

def readData(path):
    trainexamples = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            fea = []
            [fea.append(float(i)) for i in line[:line.rindex(',')].split(',')]
            label = line[line.rindex(',') + 1:].strip('\n')
            trainexamples.append(inputExamples(np.array(fea), label))
    return trainexamples


def inputExamples(fea, lab):
    return (fea, lab)


def getLabelid(labels, labels2id):
    labelsid = []
    for lab in labels:
        labelsid.append(labels2id[lab])
    return labelsid


def getMui(features, labels):
    labelcluster = {}
    mui = []
    for i in range(len(labels)):
        fea = np.array(features[i, :])
        lab = labels[i]
        if lab not in labelcluster.keys():
            labelcluster[lab] = fea
        else:
            labelcluster[lab] = np.vstack((labelcluster[lab], fea))
    for we in labelcluster.values():
        we = np.matrix(np.sum(we, axis=0)) / int(np.shape(we)[0])
        mui.append(we)
    return mui