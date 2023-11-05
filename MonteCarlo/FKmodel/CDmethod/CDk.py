import numpy as np
from ising1D import ising1D
import math
import matplotlib.pyplot as plt
import copy

"""
test the CDk method using Ising1D N=6 fixed
"""

N = 6
boundary = "fixed"

def expit(input: np.ndarray):
    return 1/(1 + np.exp(-input))

def decimal_to_bi(d: int):
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

#Import the Ising states
samples_d = np.genfromtxt("states50000.txt", dtype=np.int16)
samples_b = decimal_to_bi(samples_d[0]).reshape(1, N)
samples_num = len(samples_d)
for i in range(1,len(samples_d)):
    print(i)
    temp = decimal_to_bi(samples_d[i])
    samples_b = np.insert(samples_b, i, temp, axis=0)
# ADD a 1 to the front
samples_b = np.insert(samples_b, 0, 1, axis=1)

k = 5
#f = open("CD"+str(k)+"_sample500000.txt","w")
# num_hidden = 2
# while num_hidden <= 10:
for num_hidden in [2]:
    num_visible = 6
    np_rng = np.random.RandomState(1234)
    weights = np.asarray(np_rng.uniform(low=-0.1 * np.sqrt(6. / (num_hidden + num_visible)),high=0.1 * np.sqrt(6. / (num_hidden + num_visible)),size=(num_visible, num_hidden)))
    weights = np.insert(weights, 0, 0, axis = 0)
    weights = np.insert(weights, 0, 0, axis = 1)
    
    KL_old = 100
    KL = 10
    lr = 0.1
    step = 0
    while abs(KL_old - KL)/KL > 1E-6:
    #for step in range(1000):
        #CDk
        pos_hidden_activations = np.dot(samples_b, weights)
        pos_hidden_probs = expit(pos_hidden_activations)
        pos_hidden_probs[:, 0] = 1
        pos_associations = np.dot(samples_b.T, pos_hidden_probs) / samples_num
        pos_associations[0,0] = 0
        
        visible_states = copy.deepcopy(samples_b)
        for CDstep in range(k):
            hidden_activations = np.dot(visible_states, weights)
            hidden_probs = expit(hidden_activations)
            hidden_probs[:, 0] = 1
            hidden_states = (np.random.rand(hidden_probs.shape[0], hidden_probs.shape[1])<hidden_probs).astype(np.float32)
            
            visible_activations = np.dot(hidden_states, weights.T)
            visible_probs = expit(visible_activations)
            visible_probs[:, 0] = 1
            visible_states = (np.random.rand(visible_probs.shape[0], visible_probs.shape[1])<visible_probs).astype(np.float32)
        neg_hidden_activations = np.dot(visible_states, weights)
        neg_hidden_probs = expit(neg_hidden_activations)
        neg_hidden_probs[:, 0] = 1
        neg_associations = np.dot(visible_states.T, neg_hidden_probs) / samples_num
        neg_associations[0,0] = 0
        d_weights = pos_associations - neg_associations
        
        Z_lambda = 0
        for i in range(2**N):
            v = decimal_to_bi(i)
            v = np.insert(v, 0, 1)
            hidden_activations = np.dot(v, weights)
            Z_lambda += np.exp(hidden_activations[0]+np.logaddexp(0,hidden_activations[1:]).sum())
            
        prob_lambda = []
        H_lambda = 0
        for i in range(2**N):
            v = decimal_to_bi(i)
            v = np.insert(v, 0, 1)
            hidden_activations = np.dot(v, weights)
            prob_lambda += [np.exp(hidden_activations[0]+np.logaddexp(0,hidden_activations[1:]).sum())/Z_lambda]
            H_lambda -= prob_s[i] * np.log(prob_lambda[i])   
        
        KL_old = KL
        KL = H_lambda - H_s 
        print("step = ", step, "KL = ", KL)
        if(KL>KL_old): 
            lr /= 10
        d_weights = d_weights * lr
        weights = weights + d_weights
        step += 1
    # if KL > 1:
    #     continue
    # else:
    #     print("num_hidden = ", num_hidden, ", step = ", step, ", KL = ", KL, file = f)
    #     # print("hidden number:",num_hidden,", step =", k, ", KL = ", KL)
    #     #print(prob_lambda)
    #     fig, ax = plt.subplots()
    #     ax.plot(range(2**N), prob_s, "ks-")
    #     ax.plot(range(2**N), prob_lambda, "bo-")
    #     ax.set_yscale("log")
    #     ax.set_title("Prob distribution of Ising 1D(N=6, fixed)")
    #     ax.set_xlabel("config num")
    #     ax.set_ylabel("prob")
    #     plt.savefig("figs/CDk/CD"+str(k)+"_sample5000_nh"+str(num_hidden)+".png")
    #     num_hidden += 2
    
    
    
    






