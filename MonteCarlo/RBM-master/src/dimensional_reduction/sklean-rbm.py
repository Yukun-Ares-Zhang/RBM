"""
==============================================================
Restricted Boltzmann Machine features for digit classification
==============================================================

For greyscale image data where pixel values can be interpreted as degrees of
blackness on a white background, like handwritten digit recognition, the
Bernoulli Restricted Boltzmann machine model (:class:`BernoulliRBM
<sklearn.neural_network.BernoulliRBM>`) can perform effective non-linear
feature extraction.

In order to learn good latent representations from a small dataset, we
artificially generate more labeled data by perturbing the training data with
linear shifts of 1 pixel in each direction.

This example shows how to build a classification pipeline with a BernoulliRBM
feature extractor and a :class:`LogisticRegression
<sklearn.linear_model.LogisticRegression>` classifier. The hyperparameters
of the entire model (learning rate, hidden layer size, regularization)
were optimized by grid search, but the search is not reproduced here because
of runtime constraints.

Logistic regression on raw pixel values is presented for comparison. The
example shows that the features extracted by the BernoulliRBM help improve the
classification accuracy.
"""

from __future__ import print_function

print(__doc__)

# Authors: Yann N. Dauphin, Vlad Niculae, Gabriel Synnaeve
# License: BSD

import numpy as np
import matplotlib.pyplot as plt

from scipy.ndimage import convolve
from sklearn import linear_model, datasets, metrics
#from sklearn.cross_validation import train_test_split
from sklearn.neural_network import BernoulliRBM
#from sklearn.pipeline import Pipeline

from load_mydata import load_mydata 

import argparse 
parser = argparse.ArgumentParser()
parser.add_argument('-filename', default='../data/isingconfig_T2.5.dat')
parser.add_argument('-n_hidden', type=int, default=16)
args = parser.parse_args()


###############################################################################
# Setting up
X_train, y_train, X_test, y_test = load_mydata(args.filename) 

# Models we will use
#logistic = linear_model.LogisticRegression()
rbm = BernoulliRBM(random_state=0, verbose=True)
#classifier = Pipeline(steps=[('rbm', rbm), ('logistic', logistic)])

###############################################################################
# Training

# Hyper-parameters. These were set by cross-validation,
# using a GridSearchCV. Here we are not performing cross-validation to
# save time.
rbm.learning_rate = 0.05
rbm.n_iter = 20
# More components tend to give better prediction performance, but larger
# fitting time
rbm.n_components = args.n_hidden 
#logistic.C = 6000.0

rbm.fit(X_train)

# Training RBM-Logistic Pipeline
#classifier.fit(X_train, Y_train)

# Training Logistic regression
#logistic_classifier = linear_model.LogisticRegression(C=100.0)
#logistic_classifier.fit(X_train, Y_train)

###############################################################################
# Evaluation

#print()
#print("Logistic regression using RBM features:\n%s\n" % (
#    metrics.classification_report(
#        Y_test,
#        classifier.predict(X_test))))

#print("Logistic regression using raw pixel features:\n%s\n" % (
#    metrics.classification_report(
#        Y_test,
#        logistic_classifier.predict(X_test))))

###############################################################################
# Plotting

M = int(np.sqrt(rbm.n_components))
L = int(np.sqrt(X_train.shape[1]))

plt.figure(figsize=(4.2, 4))
for i, comp in enumerate(rbm.components_):
    plt.subplot(M, M, i + 1)
    im = plt.imshow(comp.reshape((L, L)), cmap=plt.cm.gray_r,
               interpolation='nearest')
    plt.xticks(())
    plt.yticks(())
    plt.colorbar(im) 
plt.suptitle('%g components extracted by RBM'%(rbm.n_components), fontsize=16)
plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)

plt.show()
