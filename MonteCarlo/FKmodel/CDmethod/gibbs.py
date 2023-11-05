"""
Gibbs sampling test.

Target distribution: the distribution of the RBM pre-trained using gradient descend method(KL-divergence strictly calculated, num_hidden = 2)(for details about the pre-training, refer to "GD.py")

The test varies the sampling start point k and sampling interval r and evaluate the result using Chi-square test.
"""

import numpy as np
from ising1D import ising1D
import math
import matplotlib.pyplot as plt

N = 6
boundary = "fixed"

def decimal_to_bi(d: int):
    """
    Convert decimal index to binary states.
    
    input: decimal index (0~2**N-1)
    
    output: np.ndarray, len = N vector, each entry be one of {-1, 1}
    
    indexing rule example:
    
    [1,-1,1,-1,-1,-1]-->1*2^0 + 1*2^2=5
    """ 
    assert d>=0 and d<2**N,"Invalid index number."
    temp = bin(d).replace("0b","")
    b = []
    for i in range(N):
        if i<len(temp):
            b += [int(temp[len(temp)-i-1])] 
        else:
            b += [0]
    b = np.asarray(b)
    return b

ising = ising1D(6, 1, 0, 1, "fixed")

prob_s = []
H_s = 0
for i in range(64):
    new_s = decimal_to_bi(i) * 2 - 1
    ising.modify_s(new_s)
    prob_s += [ising.prob()]
    H_s -= prob_s[i] * math.log(prob_s[i])

for num_hidden in [2]:
    num_visible = 6
    np_rng = np.random.RandomState(1234)
    weights = np.asarray(np_rng.uniform(low=-0.1 * np.sqrt(6. / (num_hidden + num_visible)),high=0.1 * np.sqrt(6. / (num_hidden + num_visible)),size=(num_visible, num_hidden)))
    weights = np.insert(weights, 0, 0, axis = 0)
    weights = np.insert(weights, 0, 0, axis = 1)

    KL_old = 100
    KL = 10
    lr = 0.1
    k = 0
    while abs(KL_old - KL)/KL > 1E-6:
        Z_lambda = 0
        for i in range(2**N):
            v = decimal_to_bi(i)
            v = np.insert(v, 0, 1)
            hidden_activations = np.dot(v, weights)
            Z_lambda += np.exp(hidden_activations[0]+np.logaddexp(0,hidden_activations[1:]).sum())
        prob_lambda = []
        H_lambda = 0
        d_weights = np.zeros([num_visible + 1, num_hidden + 1])
        for i in range(2**N):
            v = decimal_to_bi(i)
            v = np.insert(v, 0, 1)
            hidden_activations = np.dot(v, weights)
            prob_lambda += [np.exp(hidden_activations[0]+np.logaddexp(0,hidden_activations[1:]).sum())/Z_lambda]
            H_lambda -= prob_s[i] * np.log(prob_lambda[i])
            hidden_activations[0] = 1
            d_weights += np.dot(v.reshape(num_visible + 1, 1), hidden_activations.reshape(1, num_hidden + 1)) * (prob_s[i] - prob_lambda[i])
        KL_old = KL
        KL = H_lambda - H_s 
        if(KL>KL_old): 
            lr /= 10
        d_weights = d_weights * lr
        d_weights[0,0] = 0
        #print(prob_lambda)
        #print("step = ", k, "KL = ", KL)
        #print(weights)
        weights = weights + d_weights
        k += 1
    print("hidden number:",num_hidden,", step =", k, ", KL = ", KL)
   
prob_lambda = np.asarray(prob_lambda)

def expit(input: np.ndarray):
    return 1/(1 + np.exp(-input))

def gibbs_step(input: np.ndarray, weights: np.ndarray):
    """
    Perform a gibbs step.
    
    input: (1, visible vector)_k
    
    output: (1, visible vector)_{k+1}
    """
    hidden_activations = np.dot(input, weights)
    hidden_probs = expit(hidden_activations)
    hidden_probs[0] = 1
    hidden_states = (hidden_probs > np.random.rand(len(hidden_probs))).astype(np.float32)

    visible_activations = np.dot(hidden_states, weights.T)
    visible_probs = expit(visible_activations)
    visible_probs[0] = 1
    visible_states = (visible_probs > np.random.rand(len(visible_probs))).astype(np.float32)
    return visible_states

def bi_to_decimal(input: np.ndarray):
    """
    Convert binary vector to decimal int
    
    input: (1, visible vector)
    
    output: decimal(visible vector)
    
    [1, 1, 0] = 1*2^0 + 1*2^1  
    """
    s=0 
    for i in range(1, len(input)):
        s += input[i] * (2**(i-1))
    return int(s)

num_samples = 5000
test_num = 20
f = open("bin.txt","w")
for start_k in [10, 100, 1000, 10000, 100000]:
    print("k=", start_k, "\n------------")
    print("start k =", start_k,"\n","-------------------------------------------", file = f)
    for sample_interval in range(1, 11):
        print("sample interval = ", sample_interval)
        X_result = []
        for test in range(test_num):
            exact_lambda = prob_lambda * num_samples
            gibbs_lambda = np.zeros(64)

            v = np.random.randint(0, 2, num_visible)
            v = np.insert(v, 0, 1)
            for i in range(num_samples * sample_interval + start_k):
                v = gibbs_step(v, weights)
                if i>=start_k and (i-start_k)%sample_interval==0:
                    sample_result = bi_to_decimal(v)
                    gibbs_lambda[sample_result] += 1
                
            X = 0   
            for i in range(len(exact_lambda)):
                X += (gibbs_lambda[i] - exact_lambda[i])**2 / exact_lambda[i]
            X_result += [X]
        X_result = np.asarray(X_result)
        print("sample interval = ", sample_interval, "\tmean_X = ", X_result.mean(), "\tstdVar_X = ", math.sqrt(X_result.var(ddof=1)), file = f)
# fig, ax = plt.subplots()      
# ax.plot(range(2**N), exact_lambda, "ks-")
# ax.plot(range(2**N), gibbs_lambda, "bo-")
# ax.set_yscale("log")
# ax.set_title("Gibbs sampling result, X2 test: X=")
# ax.set_xlabel("config num")
# ax.set_ylabel("count")
# fig.savefig("figs/Gibbs_sample_k10_t5000_i1.png")

        