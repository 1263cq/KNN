"""
获得文件矩阵

"""
import numpy as np
import KNN02 as K2
import KNN03 as K3



def classify(str):
    num=0
    if str == 'largeDoses':
        num=3
    if str == 'smallDoses':
        num=2
    if str == 'didntLike':
        num=1
    return num

def filematrix(filename):
    with open(filename) as f:
        filelist=f.readlines()
    filelen = len(filelist)
    for i in range(filelen):
        #  去除str前后空白字符
        filelist[i] = filelist[i].strip()
        filelist[i] = filelist[i].split('\t')
        filelist[i][-1]=classify(filelist[i][-1])
    filematrix = np.matrix(filelist)
    return filematrix

if  __name__=='__main__':
    filematrix= filematrix('datingTestSet.txt')
    num = filematrix.shape[0]
    test = np.zeros((num,3))
    test = filematrix[:,0:3]
    normat = K2.normalization(test)
    #  print(normat)
    K3.showdata(normat,filematrix[:,[3]])
