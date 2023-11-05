import numpy as np 

def load_data(filename="isingconfig.dat"):
    data = np.loadtxt(filename)
    #random shuffle rows 
    np.random.shuffle(data)
    X = data[:, :-1]
    if (X.min()==-1):
        X = (X+1)/2 # \pm 1 to 0 and 1
    Y = data[:, -1]
    
    Nsample = X.shape[0]
    Ninput = X.shape[1]
    
    Ntraining = int(Nsample*0.8)

    X_train = X[:Ntraining] 
    y_train = Y[:Ntraining]

    X_test = X[Ntraining:]
    y_test = Y[Ntraining:]
    
    return X_train, y_train, X_test, y_test 
