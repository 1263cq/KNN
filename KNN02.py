import numpy as np


#  单个进行归一
def example(fm):
    fm_min = fm.astype(float).min()
    min_mat = np.tile(fm_min, (fm.shape[0], 1))
    fm_max = fm.astype(float).max()
    #   max_mat = np.tile(fm_max,(fm.shape[0],1))
    fm_range = fm_max - fm_min
    range_mat = np.tile(fm_range, (fm.shape[0], 1))
    normatrix = (fm.astype(float)-fm_min) / range_mat
    return normatrix


#  特征矩阵归一化处理
def normalization(filematrix):
    len = filematrix.shape[1]
    normat = np.zeros(np.shape(filematrix))
    for i in range(len):
        fm = filematrix[:, [i]]
        normat[:, [i]] = example(fm)
    return normat
