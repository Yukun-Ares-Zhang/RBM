import numpy as np 
import math
import torch
from torch import nn
import torch.nn.functional as F
from ising1D import ising1D

def bi_to_decimal(x):
    decimal = 0
    for i in range(6):
        if x[i]==-1:
            decimal += 2**i
    return decimal

class RBM(nn.Module):
    def __init__(self,
               n_vis=64,
               n_hin=100,
               lr=10E-3):
        super(RBM, self).__init__()
        self.W = torch.randn(n_hin, n_vis)*1e-2
        self.h_bias = torch.zeros(n_hin)
        self.v_bias = torch.zeros(n_vis)
        self.n_vis = n_vis
        self.n_hin = n_hin
        self.lr=lr
    
    def sample_from_p(self,p):
       return F.relu(torch.sign(p - (torch.rand(p.size()))))
    
    def v_to_h(self,v):
        p_h = torch.sigmoid(F.linear(v,self.W,self.h_bias))
        sample_h = self.sample_from_p(p_h)
        return p_h,sample_h
    
    def h_to_v(self,h):
        p_v = torch.sigmoid(F.linear(h,self.W.t(),self.v_bias))
        sample_v = self.sample_from_p(p_v)
        return p_v,sample_v
        
    def CDk(self,v,k):
        vk=v.clone()
        for i in range(k):
            ph, h=self.v_to_h(vk)
            pv, vk=self.h_to_v(h)
        return vk
    
    def conditional_p_h_given_v(self, v):
        """
        Calculate P(hi=1|v)
        
        input: visible units (tensor)
        
        output: P(hi=1|v)=1/(1+exp(-(h_bias+W.v))) a tensor in the same shape as hidden units.
        """
        P=F.linear(v, self.W, self.h_bias).sigmoid()
        return P
    
    def free_energy(self,v):
        """
        Calculate the free energy of a configuration.
        
        input: visible units tensor, 1D required
        
        output: E(v)=-v.v_bias-ln(1+exp(h_bias+v.W)), 0D tensor
        """
        vbias_term = torch.matmul(v,self.v_bias)
        wx_b = F.linear(v,self.W,self.h_bias)
        hidden_term = wx_b.exp().add(1).log().sum()
        return (- hidden_term - vbias_term).item()
    
    def partition_func(self, model):
        Z = 0
        for i in range(2**model.N):
            v=model.s.clone()
            model.flip()
            Z += math.exp(-self.free_energy(v))
        return Z
    
    def prob_app(self, v, model):
        Z = self.partition_func(model)
        p = math.exp(-self.free_energy(v)) / Z
        return p
    
    def step(self, model,k):
        """
        One training step for 1D Ising model. The method supposed to sample from original distribution (suggested by the model),
        but because of the small dimention, it is easier if we take all the case and their possibility. It will be exact compared to sampling.
        
        input: model (ising1D class type)
        
        output: no output. Change the self.W, self.v_bias, self.h_bias
        
        W_{ij} <-- W_{ij} + lr * \sum P(v)(P(h_i=1|v)v_j - P(h_i=1|v(k))v(k)_j)
        v_bias_{j} <-- v_bias_{j} + lr * \sum P(v)(v_j - v(k)_j)
        h_bias_{i} <-- h_bias_{i} + lr * \sum P(v)(P(h_i=1|v) - P(h_i=1|v(k)))
        """
        dW=torch.zeros(self.W.size())
        dv_bias=torch.zeros(self.v_bias.size())
        dh_bias=torch.zeros(self.h_bias.size())
        for i in range(2**model.N):
            v=model.s.clone()
            model.flip()
            vk=self.CDk(v,k)
            prob=model.prob()
            dW+=prob*(torch.matmul(self.conditional_p_h_given_v(v).reshape(-1,1),v.reshape(1,-1))\
                -torch.matmul(self.conditional_p_h_given_v(vk).reshape(-1,1),vk.reshape(1,-1)))
            dv_bias+=prob*(v-vk)
            dh_bias+=prob*(self.conditional_p_h_given_v(v)-self.conditional_p_h_given_v(vk))
            
        self.W += self.lr * dW
        self.v_bias += self.lr * dv_bias
        self.h_bias += self.lr * dh_bias
        
    def KL_divergence(self, model):
        """
        Calculate the KL_divergence between the RBM and the exact model (named model).
        
        To avoid recalculation of the partition functions:
        
        KL = \sum (all config v) p_model(v) log(p_model(v)/p_RBM(v))
        
           = \sum (all config v) exp(-E_model(v)) / Z_model * (E_RBM(v) - E_model(v) + log Z_RBM - log Z_model)
        """
        Z_RBM = self.partition_func(model)
        KL = 0
        for i in range(2**model.N):
            v = model.s.clone()
            E_RBM = self.free_energy(v)
            E_model = model.Hamitonian()
            KL += math.exp(-E_model)/model.partition_func * (E_RBM - E_model - math.log(model.partition_func) + math.log(Z_RBM))
            model.flip()
        return KL
    
model = ising1D(6,1,0,1)
rbm = RBM(6, 10, 10E-3)

for i in range(1000):
    print(rbm.KL_divergence(model))
    rbm.step(model,5)