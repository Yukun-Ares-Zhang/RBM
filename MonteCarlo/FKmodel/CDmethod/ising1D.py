'''
Calculate the exact prob distribution of 1D Ising model
'''

import numpy as np 
import math
import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

def bi_to_decimal(x):
    """
    Convert binary states to decimal index.
    
    input: array-like, len = 6 vector, each entry be one of {-1, 1}
    
    output: decimal index (0~63)
    
    indexing rule example:
    
    [1,-1,1,-1,-1,-1]-->1*2^0 + 1*2^2=5
    """ 
    decimal = 0
    for i in range(6):
        if x[i]==-1:
            decimal += 2**i
    return decimal
    
class ising1D:
    def __init__(self, N=6, J=1, uH=0, ini_mode=1, boundary = "recurrent"):
        # cases with magnetic field not included
        self.N=N
        self.J=J
        self.uH=uH
        self.boundary = boundary
        self.s=torch.ones(self.N)
        #self.partition_func = (2*math.cosh(self.J))**self.N+(2*math.sinh(self.J))**self.N
        if(ini_mode==-1):
            self.s-=2
        if(ini_mode==0):
            self.s -= 0.5
            self.s=torch.bernoulli(self.s)
        partition_func = 0
        for i in range(2**self.N):
            partition_func += math.exp(-self.Hamitonian())
            self.flip()
        self.partition_func = partition_func
    
    def Hamitonian(self):
        # Ising model: H(v)=-J \sum_<ij>{v_i v_j} - uH \sum_{i}{v_i}
        # boundary = "recurrent"-->recurrent boundary condition, v_1 v_n are adjacent
        # boundary = "fixed"-->fixed boundary condition, v_1 v_n not adjacent
        hamitonian = 0
        for i in range(self.N):
            if i!=self.N-1:
                hamitonian+=(self.s[i]*self.s[i+1]).item()
            else:
                if self.boundary == "recurrent":
                    hamitonian+=(self.s[0]*self.s[self.N-1]).item()
                if self.boundary == "fixed":
                    continue
        hamitonian*=-self.J
        return hamitonian
    
    def prob(self):
        # Calculate the probability of the configuration self.s
        return math.exp(-self.Hamitonian()) / self.partition_func
    
    def flip(self):
        # Flip the self.s 
        # Flipping rule: similar to binary addition
        # eg. 
        # [-1,1,-1,-1,-1,-1]-->[1,1,-1,-1,-1,-1]
        # [1,1,-1,1,-1,1]-->[-1,-1,1,1,-1,1]
        index=0
        while(index<self.N and self.s[index].item()==-1):
            index+=1
        if(index==self.N):
            self.s=-self.s
        else:
            self.s[:(index+1)]=-self.s[:(index+1)]
            
    def initialize(self):
        self.s=torch.ones(self.N)
        
    def to_decimal(self):
        """
        Convert binary states to decimal index.
        
        input: array-like, len = 6 vector, each entry be one of {-1, 1}
        
        output: decimal index (0~63)
        
        indexing rule example:
        
        [1,-1,1,-1,-1,-1]-->1*2^0 + 1*2^2=5
        """ 
        decimal = 0
        for i in range(self.N):
            if self.s[i] == -1:
                decimal += 2**i
        return decimal
    
    def modify_s(self, new_s):
        # Change the spin state self.s
        assert len(new_s) == self.N, "Vector length not match."
        s = torch.tensor(new_s).to(torch.float32)
        self.s = s
        

model = ising1D(6,1,0,1,"fixed")
config = []
prob = []
for i in range(2**6):
    config += [model.to_decimal()]
    prob += [model.prob()]
    model.flip()

fig, ax = subplots()    
ax.plot(config, prob, "ks-")
ax.set_yscale("log")
ax.set_title("Prob distribution of Ising 1D(N=6, fixed)")
ax.set_xlabel("config num")
ax.set_ylabel("prob")
fig.savefig("figs/show.png")
