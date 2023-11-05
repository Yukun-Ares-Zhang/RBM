"""Restricted Boltzmann Machine
"""

# Authors: Yann N. Dauphin <dauphiya@iro.umontreal.ca>
#          Vlad Niculae
#          Gabriel Synnaeve
#          Lars Buitinck
# License: BSD 3 clause

import time

import numpy as np
import scipy.sparse as sp

from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.externals.six.moves import xrange
from sklearn.utils import check_array
from sklearn.utils import gen_even_slices
from sklearn.utils import issparse
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.utils.extmath import log_logistic
from sklearn.utils.fixes import expit             # logistic function
from sklearn.utils.validation import check_is_fitted
import h5py 

class BernoulliRBM(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=256, resfolder='./'):
        self.n_components = n_components
        self.resfolder = resfolder

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
                - np.logaddexp(0, safe_sparse_dot(v, self.components_.T)
                               + self.intercept_hidden_).sum(axis=1))

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

    #def _energy(self, h, v):
    #    return (-safe_sparse_dot(h, self.intercept_hidden_) 
    #            -safe_sparse_dot(v, self.intercept_visible_)
    #            -safe_sparse_dot(safe_sparse_dot(h, self.components_), v.T) 
    #            )

    def set_weights(self, W, b, a):
        # W.shape = (nv, nh) = (n_features, n_components)
        # b.shape = (nh,) = (n_components)
        # a.shape = (nv,) = (n_features)
        self.components_ = W.T 
        self.intercept_hidden_ = b 
        self.intercept_visible_ = a #np.zeros(W.shape[0], )

if __name__=='__main__':


    X = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
    rbm = BernoulliRBM(n_components=2)
    rbm.fit(X)
    
    v = np.array([0, 0, 0])
    v.shape = (1, -1)

    print rbm._free_energy(v)

    h = np.array([0, 0])
    h.shape = (1, -1)
    print rbm._free_energy_hidden(h)

    print rbm._energy(h, v)

