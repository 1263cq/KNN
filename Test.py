import numpy as np
import KNN02 as K2
import KNN03 as K3
import KNN01 as K1
import KNN04 as K4

filematrix = K1.filematrix('datingTestSet.txt')
num = filematrix.shape[0]
test = np.zeros((num, 3))
test = filematrix[:, 0:3]
normat = K2.normalization(test)
labels = filematrix[100:,[3]].tolist()
label = filematrix[:,[3]].tolist()
flase=0
for i in range(100):
    result=K4.classify0(normat[[i],:],normat[100:,:],labels,4)

    if result !=str(label[i]):
        flase+=1
error=flase/100
print('错误率为%.2f' % error)


#  print(normat)
#  K3.showdata(normat, filematrix[:, [3]])
