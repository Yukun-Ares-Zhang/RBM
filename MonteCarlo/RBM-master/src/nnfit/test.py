import h5py 
import numpy as np 

h5 = h5py.File('../../data/nnfit/L8T2.2691853nhidden256/weights.hdf5','r')

W = np.array(h5['dense_1/dense_1_W'][()])
b = np.array(h5['dense_1/dense_1_b'][()])

def f(x): 
    return (np.log(1.+np.exp(b+np.dot(x,W)))).sum()

x = np.zeros(64)
x[2] = 1
#x = np.random.randint(2, size=64)
#print x.shape 
#print np.dot(x,W).shape 
#print np.exp(b+np.dot(x,W)).shape 

print x 
print f(x)
