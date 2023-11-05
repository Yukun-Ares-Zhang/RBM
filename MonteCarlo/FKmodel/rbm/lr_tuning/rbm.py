"""Restricted Boltzmann Machine
"""

# Authors: Yann N. Dauphin <dauphiya@iro.umontreal.ca>
#          Vlad Niculae
#          Gabriel Synnaeve
#          Lars Buitinck
# License: BSD 3 clause

import time
import os
import matplotlib.pyplot as plt

import numpy as np
np.set_printoptions(threshold=np.inf)
import scipy.sparse as sp

from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.externals.six.moves import xrange
from sklearn.utils import check_array
from sklearn.utils import check_random_state
from sklearn.utils import gen_even_slices
from sklearn.utils import issparse
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.utils.extmath import log_logistic
from scipy.special import expit             # logistic function
from sklearn.utils.validation import check_is_fitted
import h5py 

#for plotting 
try:
    import PIL.Image as Image
except ImportError:
    import Image
from utils import tile_raster_images


class BernoulliRBM(BaseEstimator, TransformerMixin):
    """Bernoulli Restricted Boltzmann Machine (RBM).

    A Restricted Boltzmann Machine with binary visible units and
    binary hiddens. Parameters are estimated using Stochastic Maximum
    Likelihood (SML), also known as Persistent Contrastive Divergence (PCD)
    [2].

    The time complexity of this implementation is ``O(d ** 2)`` assuming
    d ~ n_features ~ n_components.

    Read more in the :ref:`User Guide <rbm>`.

    Parameters
    ----------
    n_components : int, optional
        Number of binary hidden units.

    learning_rate : float, optional
        The learning rate for weight updates. It is *highly* recommended
        to tune this hyper-parameter. Reasonable values are in the
        10**[0., -3.] range.

    batch_size : int, optional
        Number of examples per minibatch.

    n_iter : int, optional
        Number of iterations/sweeps over the training dataset to perform
        during training.

    verbose : int, optional
        The verbosity level. The default, zero, means silent mode.

    random_state : integer or numpy.RandomState, optional
        A random number generator instance to define the state of the
        random permutations generator. If an integer is given, it fixes the
        seed. Defaults to the global numpy random number generator.

    Attributes
    ----------
    intercept_hidden_ : array-like, shape (n_components,)
        Biases of the hidden units.

    intercept_visible_ : array-like, shape (n_features,)
        Biases of the visible units.

    components_ : array-like, shape (n_components, n_features)
        Weight matrix, where n_features in the number of
        visible units and n_components is the number of hidden units.

    Examples
    --------

    >>> import numpy as np
    >>> from sklearn.neural_network import BernoulliRBM
    >>> X = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
    >>> model = BernoulliRBM(n_components=2)
    >>> model.fit(X)
    BernoulliRBM(batch_size=10, learning_rate=0.1, n_components=2, n_iter=10,
           random_state=None, verbose=0)

    References
    ----------

    [1] Hinton, G. E., Osindero, S. and Teh, Y. A fast learning algorithm for
        deep belief nets. Neural Computation 18, pp 1527-1554.
        http://www.cs.toronto.edu/~hinton/absps/fastnc.pdf

    [2] Tieleman, T. Training Restricted Boltzmann Machines using
        Approximations to the Likelihood Gradient. International Conference
        on Machine Learning (ICML) 2008
    """
    def __init__(self, n_components=256, learning_rate=0.1, batch_size=10,
                 n_iter=10, verbose=0, random_state=None, resfolder='./'):
        self.n_components = n_components
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_iter = n_iter
        self.verbose = verbose
        self.random_state = random_state
        self.resfolder = resfolder

    def transform(self, X):
        """Compute the hidden layer activation probabilities, P(h=1|v=X).

        Parameters
        ----------
        X : {array-like, sparse matrix} shape (n_samples, n_features)
            The data to be transformed.

        Returns
        -------
        h : array, shape (n_samples, n_components)
            Latent representations of the data.
        """
        check_is_fitted(self, "components_")

        X = check_array(X, accept_sparse='csr', dtype=np.float)
        return self._mean_hiddens(X)

    def _mean_hiddens(self, v):
        """Computes the probabilities P(h=1|v).

        Parameters
        ----------
        v : array-like, shape (n_samples, n_features)
            Values of the visible layer.

        Returns
        -------
        h : array-like, shape (n_samples, n_components)
            Corresponding mean field values for the hidden layer.
        """
        p = safe_sparse_dot(v, self.components_.T)
        p += self.intercept_hidden_
        return expit(p, out=p)

    def _sample_hiddens(self, v, rng):
        """Sample from the distribution P(h|v).

        Parameters
        ----------
        v : array-like, shape (n_samples, n_features)
            Values of the visible layer to sample from.

        rng : RandomState
            Random number generator to use.

        Returns
        -------
        h : array-like, shape (n_samples, n_components)
            Values of the hidden layer.
        """
        p = self._mean_hiddens(v)
        return (rng.random_sample(size=p.shape) < p)

    def _sample_visibles(self, h, rng):
        """Sample from the distribution P(v|h).

        Parameters
        ----------
        h : array-like, shape (n_samples, n_components)
            Values of the hidden layer to sample from.

        rng : RandomState
            Random number generator to use.

        Returns
        -------
        v : array-like, shape (n_samples, n_features)
            Values of the visible layer.
        """
        p = np.dot(h, self.components_)
        p += self.intercept_visible_
        expit(p, out=p)
        return (rng.random_sample(size=p.shape) < p)

    def _free_energy(self, v):
        """Computes the free energy F(v) = - log sum_h exp(-E(v,h)).

        Parameters
        ----------
        v : array-like, shape (n_samples, n_features)
            Values of the visible layer.

        Returns
        -------
        free_energy : array-like, shape (n_samples,)
            The value of the free energy.
        """
        return (- safe_sparse_dot(v, self.intercept_visible_)
                - np.logaddexp(0, safe_sparse_dot(v, self.components_.T) + self.intercept_hidden_).sum(axis=1))

    def _free_energy_hidden(self, h):
        """Computes the free energy F(h) = - log sum_v exp(-E(v,h)).

        Parameters
        ----------
        h : array-like, shape (n_samples, n_components)
            Values of the visible layer.

        Returns
        -------
        free_energy : array-like, shape (n_samples,)
            The value of the free energy.
        """
        return (- safe_sparse_dot(h, self.intercept_hidden_)
                - np.logaddexp(0, safe_sparse_dot(h, self.components_)
                               + self.intercept_visible_).sum(axis=1))


    def _energy(self, h, v):
        return (-safe_sparse_dot(h, self.intercept_hidden_) 
                -safe_sparse_dot(v, self.intercept_visible_)
                -safe_sparse_dot(safe_sparse_dot(h, self.components_), v.T) 
                )

    def gibbs(self, v):
        """Perform one Gibbs sampling step.

        Parameters
        ----------
        v : array-like, shape (n_samples, n_features)
            Values of the visible layer to start from.

        Returns
        -------
        v_new : array-like, shape (n_samples, n_features)
            Values of the visible layer after one Gibbs step.
        """
        check_is_fitted(self, "components_")
        if not hasattr(self, "random_state_"):
            self.random_state_ = check_random_state(self.random_state)
        h_ = self._sample_hiddens(v, self.random_state_)
        v_ = self._sample_visibles(h_, self.random_state_)

        return v_

    def partial_fit(self, X, y=None):
        """Fit the model to the data X which should contain a partial
        segment of the data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data.

        Returns
        -------
        self : BernoulliRBM
            The fitted model.
        """
        X = check_array(X, accept_sparse='csr', dtype=np.float)
        if not hasattr(self, 'random_state_'):
            self.random_state_ = check_random_state(self.random_state)
        if not hasattr(self, 'components_'):
            self.components_ = np.asarray(
                self.random_state_.normal(
                    0,
                    0.01,
                    (self.n_components, X.shape[1])
                ),
                order='F')
        if not hasattr(self, 'intercept_hidden_'):
            self.intercept_hidden_ = np.zeros(self.n_components, )
        if not hasattr(self, 'intercept_visible_'):
            self.intercept_visible_ = np.zeros(X.shape[1], )
        if not hasattr(self, 'h_samples_'):
            self.h_samples_ = np.zeros((self.batch_size, self.n_components))

        self._fit(X, self.random_state_)

    def _fit(self, v_pos, rng):
        """Inner fit for one mini-batch.

        Adjust the parameters to maximize the likelihood of v using
        Stochastic Maximum Likelihood (SML).

        Parameters
        ----------
        v_pos : array-like, shape (n_samples, n_features)
            The data to use for training.

        rng : RandomState
            Random number generator to use for sampling.
        """
        h_pos = self._mean_hiddens(v_pos)
        v_neg = self._sample_visibles(self.h_samples_, rng)
        h_neg = self._mean_hiddens(v_neg)

        lr = float(self.learning_rate) / v_pos.shape[0]
        update = safe_sparse_dot(v_pos.T, h_pos, dense_output=True).T
        update -= np.dot(h_neg.T, v_neg)
        self.components_ += lr * update
        self.intercept_hidden_ += lr * (h_pos.sum(axis=0) - h_neg.sum(axis=0))
        self.intercept_visible_ += lr * (np.asarray(
                                         v_pos.sum(axis=0)).squeeze() -
                                         v_neg.sum(axis=0))

        h_neg[rng.uniform(size=h_neg.shape) < h_neg] = 1.0  # sample binomial
        self.h_samples_ = np.floor(h_neg, h_neg)

    def score_samples(self, X):
        """Compute the pseudo-likelihood of X.

        Parameters
        ----------
        X : {array-like, sparse matrix} shape (n_samples, n_features)
            Values of the visible layer. Must be all-boolean (not checked).

        Returns
        -------
        pseudo_likelihood : array-like, shape (n_samples,)
            Value of the pseudo-likelihood (proxy for likelihood).

        Notes
        -----
        This method is not deterministic: it computes a quantity called the
        free energy on X, then on a randomly corrupted version of X, and
        returns the log of the logistic function of the difference.
        """
        check_is_fitted(self, "components_")

        v = check_array(X, accept_sparse='csr')
        rng = check_random_state(self.random_state)

        # Randomly corrupt one feature in each sample in v.
        ind = (np.arange(v.shape[0]),
               rng.randint(0, v.shape[1], v.shape[0]))
        if issparse(v):
            data = -2 * v[ind] + 1
            v_ = v + sp.csr_matrix((data.A.ravel(), ind), shape=v.shape)
        else:
            v_ = v.copy()
            v_[ind] = 1 - v_[ind]

        fe = self._free_energy(v)
        fe_ = self._free_energy(v_)
        return v.shape[1] * log_logistic(fe_ - fe)

    def load(self, model, n_features):

        h5 = h5py.File(model,'r')
        try:
            W = np.array(h5['mylayer_1/mylayer_1_W:0'][()])
            b = np.array(h5['mylayer_1/mylayer_1_b:0'][()])
        except KeyError:
            W = np.array(h5['mylayer_1/mylayer_1_W'][()])
            b = np.array(h5['mylayer_1/mylayer_1_b'][()])
        h5.close() 

        self.components_ = W.T 
        self.intercept_hidden_ = b 
        self.intercept_visible_ = np.zeros(n_features, )

        print(self.components_.shape) 
        print(self.intercept_hidden_.shape) 

    def fit(self, X, y=None):
        """Fit the model to the data X.

        Parameters
        ----------
        X : {array-like, sparse matrix} shape (n_samples, n_features)
            Training data.

        Returns
        -------
        self : BernoulliRBM
            The fitted model.
        """
        X = check_array(X, accept_sparse='csr', dtype=np.float)
        n_samples = X.shape[0]
        rng = check_random_state(self.random_state)

        self.components_ = np.asarray(
            rng.normal(0, 0.01, (self.n_components, X.shape[1])),
            order='F')
        self.intercept_hidden_ = np.zeros(self.n_components, )
        self.intercept_visible_ = np.zeros(X.shape[1], )
        self.h_samples_ = np.zeros((self.batch_size, self.n_components))

        n_batches = int(np.ceil(float(n_samples) / self.batch_size))
        batch_slices = list(gen_even_slices(n_batches * self.batch_size,
                                            n_batches, n_samples))
        verbose = self.verbose
        begin = time.time()
        for iteration in xrange(1, self.n_iter + 1):
            for batch_slice in batch_slices:
                self._fit(X[batch_slice], rng)

            if verbose:
                end = time.time()
                print("[%s] Iteration %d, pseudo-likelihood = %.2f,"
                      " time = %.2fs"
                      % (type(self).__name__, iteration,
                         self.score_samples(X).mean(), end - begin))
                begin = end

                L = int(np.sqrt(X.shape[1]))
                M = int(np.sqrt(self.n_components))
                # Construct image from the weight matrix
                image = Image.fromarray(
                    tile_raster_images(
                        X=self.components_,
                        img_shape=(L, L),
                        tile_shape=(M, M),
                        tile_spacing=(1, 1)
                    )
                )
                image.save(self.resfolder+'/filters_at_epoch_%i.png' % iteration)

        return self


if __name__=='__main__':

    total_reftime = time.time()
    X = np.load("../data_gen/Traning_data.npy")
    Training_v = X[:, 0:-1]
    Training_erg = X[:, -1]
    Y = np.load("../data_gen/Test_data.npy")
    Test_v = Y[:, 0:-1]
    Test_erg = Y[:, -1]
    
    for lr in [0.07,0.06,0.05,0.04,0.03,0.02,0.01,\
        0.009,0.008,0.007,0.006,0.005,0.004,0.003,0.002,0.001,\
        0.0009,0.0008,0.0007,0.0006,0.0005,0.0004,0.0003,0.0002,0.0001]:
        reftime = time.time()
        if os.path.exists("data")==False:
            os.makedirs("data")
        os.chdir("data")
        if os.path.exists("lr="+str(lr))==False:
            os.makedirs("lr="+str(lr))
        os.chdir("lr="+str(lr))
        
        for n_iteration in [100, 1000]: 
            if os.path.exists("n="+str(n_iteration))==False:
                os.makedirs("n="+str(n_iteration))
            os.chdir("n="+str(n_iteration))   
            rbm = BernoulliRBM(n_components=100, learning_rate=lr, batch_size=10, n_iter=n_iteration, random_state=42, verbose=0)
            rbm.fit(Training_v)
            
            Predict_erg = -rbm._free_energy(Test_v)
            Test_erg -= Test_erg.min()
            Predict_erg -= Predict_erg.min()
            erg_tuple = []
            for i in range(Test_erg.shape[0]):
                erg_tuple += [(Predict_erg[i], Test_erg[i])]
            erg_tuple = sorted(erg_tuple, key=lambda x:x[1], reverse=True)
            with open("./energy.txt", "w") as f1:
                print("Pred\tExac", file = f1)
                for i in range(Test_erg.shape[0]):
                    print("%5.2f\t%5.2f"%(erg_tuple[i][0], erg_tuple[i][1]),file = f1)
            np.savetxt("weights.txt", rbm.components_, fmt="%5.6f", delimiter="\t")
            np.savetxt("intercept_hidden.txt", rbm.intercept_hidden_, fmt="%5.6f", delimiter="\t")
            np.savetxt("intercept_visible.txt", rbm.intercept_visible_, fmt="%5.6f", delimiter="\t")
            os.chdir("..")
        
        erg1=np.loadtxt("n=100/energy.txt", skiprows=1)
        erg2=np.loadtxt("n=1000/energy.txt", skiprows=1)
        os.chdir("..\\..")
        if os.path.exists("fig")==False:
            os.makedirs("fig")
        os.chdir("fig")
        
        index = np.arange(1, erg1.shape[0]+1, 1)
        fig = plt.figure()
        fig.tight_layout(h_pad=4)
        ax1 = fig.add_subplot(1,2,1)
        ax1.scatter(index, erg1[:,0], marker="o", c="b")
        ax1.scatter(index, erg1[:,1], marker="o", c="r")
        ax1.set_title("n_iteration=100")
        ax1.set_xlabel("test sample")
        ax1.set_ylabel("erg")

        ax2 = fig.add_subplot(1,2,2)
        ax2.scatter(index, erg2[:,0], marker="o", c="b")
        ax2.scatter(index, erg2[:,1], marker="o", c="r")
        ax2.set_title("n_iteration=1000")
        ax2.set_xlabel("test sample")
        ax2.set_ylabel("erg")
        
        fig.suptitle("lr="+str(lr))
        plt.savefig("lr="+str(lr)+".png")
        os.chdir("..")
        
    print("finished lr="+str(lr)+" in %5.2f min, Total time: %5.2f min"%((time.time()-reftime)/60,(time.time()-total_reftime)/60))
        
        
            
        