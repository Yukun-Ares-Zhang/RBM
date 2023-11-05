from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers import Lambda
from keras.optimizers import SGD, Adam, Adamax, RMSprop, Adagrad
from keras.regularizers import WeightRegularizer, l2
from keras.callbacks import Callback, ModelCheckpoint
from mylayer import Mylayer 
import numpy as np 

class Network:
    def __init__(self, n_visible, n_hidden):

        self.model = Sequential()
        self.model.add(Mylayer(input_dim=n_visible,
                                output_dim=1,
                                hidden_dim=n_hidden, 
                                #weights = [np.random.randn(n_visible, n_hidden)*0.001, 
                                #           np.zeros(n_hidden)] , 
                                W_regularizer=l2(0.0001)
                                #b_regularizer=l2(0.001)
                                #c_regularizer=l2(0.001)
                              ))
        self.model.summary() 

        #optm = SGD(lr=0.001, decay=1e-6, momentum=.9)
        #optm = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        optm = Adamax(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        #optm = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08)
        #optm = Adagrad(lr=0.001, epsilon=1e-08)
 
        self.model.compile(loss='mse',optimizer=optm)

    def fit(self, X, y, verbose):
        #print (X.shape)
        #print (y.shape)
        if verbose:
            self.model.fit(X, y, nb_epoch=1, batch_size=1, verbose=verbose, 
                           validation_data = (self.X_test, self.y_test)
                          )
        else:
            self.model.fit(X, y, nb_epoch=1, batch_size=1, verbose=verbose)
    
    def get_weights(self):
        W, b, a, c = self.model.get_weights()
        return W, b, a

    def load_weights(self, f):
        self.model.load_weights(f)
    
    def save_weights(self, f):
        self.model.save_weights(f, overwrite=True)


if __name__=='__main__':
    
    n_visible = 16 
    n_hidden = 4 
    net = Network(n_visible, n_hidden)

    X = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    y = np.array([0.0])
    net.fit(X, y)
    W, b = net.get_weights()
    print W.shape 
    print b.shape 
    net.save_weights('weights.h5')
