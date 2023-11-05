from __future__ import print_function
import numpy as np 
from load_data import load_data
import sys, os, re 
from keras import backend as K 
from plot_weights import plot_weights

if __name__=='__main__':

    import argparse 
    parser = argparse.ArgumentParser()
    parser.add_argument('-filename', default='isingconfig.dat')
    parser.add_argument('-Nsamples', type=int, default=50000)
    parser.add_argument('-n_hidden', type=int, default=64)
    parser.add_argument('-n_epoch', type=int, default=20)
    parser.add_argument("-resfolder",default='../../data/nnfit/' ,help="folder")
    parser.add_argument("-loadmodel", action='store_true',  help="")

    args = parser.parse_args()

    L = int(re.search('_L([0-9]*)_',args.filename).group(1)) 
    T = float(re.search('T(-?[0-9]*\.?[0-9]*)_',args.filename).group(1)) 

    resfolder = args.resfolder + '/'  \
               + os.path.basename(args.filename).replace('.dat','') \
               + 'nhidden' + str(args.n_hidden) + '_'
    
    if not os.path.isdir(resfolder):
        os.makedirs(resfolder)

    X_train, y_train, X_test, y_test = load_data(args.filename, args.Nsamples) 
    import theano
    theano.config.floatX = 'float32'
    X_train = X_train.astype(theano.config.floatX)
    X_test = X_test.astype(theano.config.floatX)

    from keras.models import Sequential
    from keras.layers.core import Dense, Activation
    from keras.layers import Lambda
    from keras.optimizers import SGD, Adam, Adamax, RMSprop, Adagrad
    from keras.regularizers import WeightRegularizer, l2
    from keras.callbacks import Callback, ModelCheckpoint
    from mylayer import Mylayer 

    np.random.seed(42)
    model = Sequential()

    model.add(Mylayer(input_dim=X_train.shape[1],
                      output_dim=1,
                      hidden_dim=args.n_hidden, 
                      weights = [np.random.randn(X_train.shape[1], args.n_hidden)*0.001, 
                                                 np.zeros(args.n_hidden)] , 
                      bias = False,
                      W_regularizer=l2(0.0001)
                      #b_regularizer=l2(0.001)
                      #c_regularizer=l2(0.001)
                    ))
    model.summary() 

    #optm = SGD(lr=0.001, decay=1e-6, momentum=.9)
    optm = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    #optm = Adamax(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    #optm = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08)
    #optm = Adagrad(lr=0.001, epsilon=1e-08)

    model.compile(loss='mse', 
                  optimizer=optm
                  #metrics=['accuracy'], 
                  )


    class PlotW(Callback):
        def on_train_begin(self, logs={}):
            pass 
        def on_epoch_end(self, epoch, logs={}):
            W = self.model.get_weights()[0]
            #print "#############"
            #print W.shape 
            #print '#############'
            plot_weights(W, epoch, resfolder)

    class LossHistory(Callback):
        def on_train_begin(self, logs={}):
            pass
        def on_epoch_end(self, epoch, logs={}):
            f = open(resfolder+'/losses.dat','a')
            f.write('%g %g %g' %(epoch, logs.get('loss'), logs.get('val_loss'))+'\n')
            f.close()

    modelfile = resfolder+"/weights.hdf5"
    if os.path.isfile(modelfile) and args.loadmodel:
        model.load_weights(modelfile)
        print ("loaded model from", modelfile)

    plotW = PlotW()
    checkpointer = ModelCheckpoint(filepath=modelfile, save_weights_only=True, verbose=0, save_best_only=True)
    history = LossHistory()
    model.fit(X_train, y_train, nb_epoch=args.n_epoch, batch_size=1, verbose=1, 
              validation_data = (X_test, y_test),
              callbacks=[plotW, checkpointer, history]
             ) 

    #score = model.evaluate(X_test, y_test, verbose=1)
    #print('Test score:', score[0])
    #print('Test accuracy:', score[1])

    y_predict = model.predict(X_test, batch_size=1, verbose=1)
    f = open(resfolder+'/fit.dat','w')
    for i in range(len(y_test)):
        #print (X_test[i], y_test[i], y_predict[i])
        f.write('%g %g %g' %(i, y_test[i], y_predict[i])+'\n')
    f.close()
