import operator

import numpy as np


def classify0(inX,dataset,labels,k):
    num=dataset.shape[0]
    diffMat = np.tile(inX,(num,1))
    sqDiff = (diffMat-dataset)**2
    sqDistan=sqDiff.sum(1)
    dis = sqDistan**0.5
    #  返回的为列表索引值，非矩阵
    index = dis.argsort()
    classcount={}
    for i in range(k):
        votellabels = str(labels[index[i]])
        classcount[votellabels] = classcount.get(votellabels,0)+1
    sortedClassCount = sorted(classcount.items(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]


