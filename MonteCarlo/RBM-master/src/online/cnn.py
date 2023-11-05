from keras.models import Sequential
from keras.layers.core import Dense, Activation, Permute 
from keras.layers import Lambda, Flatten 
from keras.layers.pooling import GlobalAveragePooling2D
from keras.layers.convolutional import Convolution2D
from keras.optimizers import SGD, Adam, Adamax, RMSprop, Adagrad
from keras.regularizers import WeightRegularizer, l2
from keras.callbacks import Callback, ModelCheckpoint
from mylayer import Mylayer 
import numpy as np 

class Network:
    def __init__(self, n_visible, n_hidden):

        self.n_visible = n_visible 
        self.n_hidden = n_hidden

        self.L = int(np.sqrt(n_visible))

        self.model = Sequential()
        #apply the same convolution to each line
        #the outcome averages over each shift copy 
        #and sums over the hidden units 
        self.model.add(Convolution2D(n_hidden, 1, n_visible, 
                       border_mode='valid',
                       input_shape=(1, n_visible, n_visible) , 
                       weights = [np.random.randn(n_hidden, 1, 1, n_visible)*0.001, 
                                  np.zeros(n_hidden)] 
                       #W_regularizer=l2(0.0001)
                      ))
        self.model.add(Activation('softplus'))
        self.model.add(Permute((3,2,1), input_shape=(n_hidden, n_visible, 1)))
        self.model.add(GlobalAveragePooling2D())
        self.model.add(Lambda(lambda x: n_hidden*x))
        self.model.summary() 

        #optm = SGD(lr=0.001, decay=1e-6, momentum=.9)
        #optm = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        #optm = Adamax(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        optm = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08)
        #optm = Adagrad(lr=0.001, epsilon=1e-08)
 
        self.model.compile(loss='mse',optimizer=optm)

    def fit(self, X, y, verbose):
        #print (X.shape)
        #print (y.shape)
        X.shape = (self.L, self.L)
        Xextend = np.zeros((1, 1, self.n_visible, self.n_visible))
        counter = 0 
        for xshift in range(self.L):
            for yshift in range(self.L):
                Xextend[0, 0, counter] = X.reshape(1, -1)
                #print xshift, yshift 
                #print X
                counter += 1 
                X = np.roll(X, 1, axis=0)
            X = np.roll(X, 1, axis=1)
        #print Xextend
        self.model.fit(Xextend, y, nb_epoch=1, batch_size=1, verbose=verbose)
    
    def get_weights(self):
        W, b = self.model.get_weights()
        W.shape = (self.n_hidden, self.n_visible) 
        return W.transpose(), b 

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
