"""
Sampling 1D ising model with N spins. 
boundary condition: fixed/recurrent
"""
import numpy as np
from ising1D import ising1D
import math
import matplotlib.pyplot as plt
import copy

sample_num = 50000
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
# Calculate the prob distribution list:: prob_s[]
for i in range(64):
    new_s = decimal_to_bi(i) * 2 - 1
    ising.modify_s(new_s)
    prob_s += [ising.prob()]


sample_result = []# Sampling result list, the decimal index of states are recorded
sample_count = np.zeros(2**N)# Sampling result count, sample_count[i] is the number of samples with index i

prob_s_cdf = [0]
# Calculate the Cumulative distribution list:: prob_s_cdf[]
# CDF list::[0, 0.2, 0.4, 1] if CDF[i]<=rand<CDF[i+1], sample result is i
for i in range(len(prob_s)):
    s = 0
    for j in range(i+1):
        s += prob_s[j]
    prob_s_cdf += [s]

f = open("states"+str(sample_num)+".txt","w")

# Sample states based on calculated CDF 
for i in range(sample_num):
    rand = np.random.rand()
    for j in range(len(prob_s_cdf)-1):
        if rand > prob_s_cdf[j] and rand < prob_s_cdf[j+1]:
            sample_result += [j]
            sample_count[j] += 1
            print(j, file=f)
            break
print(sample_count)    

# Performing the Chi-Squared Test
X=0
for i in range(len(sample_count)):
    X += (sample_count[i] - sample_num * prob_s[i])**2 / (sample_num * prob_s[i])
      
fig, ax = plt.subplots()
ax.plot(range(2**N), np.asarray(prob_s) * sample_num, "ks-")
ax.plot(range(2**N), sample_count, "bo-")
ax.set_title("Ising 1D (N=6, fixed) exact and sample PDF" + f", X2 = {X:.3f}")
ax.set_xlabel("config")
ax.set_ylabel("count")
fig.savefig("figs/show.png")