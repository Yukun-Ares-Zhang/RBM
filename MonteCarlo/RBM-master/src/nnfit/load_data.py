import numpy as np 
import h5py 
import os 
import pandas as pd 

def load_data(filename="isingconfig.dat", N=None):

    h5file = filename.replace('.dat', '.h5')
    if os.path.exists(h5file) and (os.path.getmtime(h5file)>os.path.getmtime(filename)): #loading from h5 file is more efficient 
        print ("load from hdf5 file")

        h5 = h5py.File(filename.replace('.dat', '.h5'),'r')
        data = np.array( h5['data'] ) 
        h5.close() 

    else:
        print ("load from data file")

        #data = np.loadtxt(filename)
        #equivalent ways to speed up loading data
        df = pd.read_csv(filename,sep='\s+',header=None)
        data = np.array(df)
        
        #create h5file 
        h5 = h5py.File(h5file,'w')
        h5.create_dataset('data',  data=data)
        h5.close() 

    #random shuffle rows 
    if N is not None:
        data = data[:N]
    np.random.shuffle(data)
    X = data[:, :-1]
    if (X.min()==-1):
        X = (X+1)/2 # \pm 1 to 0 and 1
    Y = data[:, -1]
    Y -= Y.min() 
    print (Y)
    
    Nsample = X.shape[0]
    Ninput = X.shape[1]
    
    Ntraining = int(Nsample*0.8)

    X_train = X[:Ntraining] 
    y_train = Y[:Ntraining]

    X_test = X[Ntraining:]
    y_test = Y[Ntraining:]
    
    return X_train, y_train, X_test, y_test 
