import numpy as np
newresult = np.arange(0,6)/1.0
newresult2 = np.arange(4,10)/1.0
res = np.array([newresult,newresult2])
print(res)
with open('test.csv','ab') as f:
    for i in range(4):
        np.savetxt(f, res.T, delimiter=",", fmt='%.4f')


arr1, arr2 = np.loadtxt('test.csv',delimiter = ",").T
print(arr1)