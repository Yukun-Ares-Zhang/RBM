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

# Calculate the exact prob distribution:: list prob_s[]
# Calculate the model Entropy H_s = -\sum p_s log{p_s}
prob_s = []
H_s = 0
for i in range(64):
    new_s = decimal_to_bi(i) * 2 - 1
    ising.modify_s(new_s)
    prob_s += [ising.prob()]
    H_s -= prob_s[i] * math.log(prob_s[i])

for num_hidden in [50]:
    num_visible = 6
    np_rng = np.random.RandomState(1234)
    """
    weight matrix:
    | 0 c.T|
    | b W  |
    Initialize a weight matrix, of dimensions (num_visible x num_hidden), using
    a uniform distribution between -sqrt(6. / (num_hidden + num_visible))
    and sqrt(6. / (num_hidden + num_visible)). One could vary the 
    standard deviation by multiplying the interval with appropriate value.
    Here we initialize the weights with mean 0 and standard deviation 0.1. 
    Reference: Understanding the difficulty of training deep feedforward 
    neural networks by Xavier Glorot and Yoshua Bengio
    """
    weights = np.asarray(np_rng.uniform(low=-0.1 * np.sqrt(6. / (num_hidden + num_visible)),high=0.1 * np.sqrt(6. / (num_hidden + num_visible)),size=(num_visible, num_hidden)))
    weights = np.insert(weights, 0, 0, axis = 0)
    weights = np.insert(weights, 0, 0, axis = 1)

    KL_old = 100
    KL = 10
    lr = 0.1
    k = 0
    # convergence criterion: KL relative renewal |KL_old - KL|/KL
    while abs(KL_old - KL)/KL > 1E-6:
        # Calculate the RBM partition function Z_\lambda
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
    #print(prob_lambda)
    plt.plot(range(2**N), prob_s, "ks-")
    plt.plot(range(2**N), prob_lambda, "bo-")
    plt.yscale("log")
    plt.title("Prob distribution of Ising 1D(N=6, fixed)")
    plt.xlabel("config num")
    plt.ylabel("prob")
    plt.savefig("figs/exact_and_trained_prob_distribution_fixed3.png")
    






