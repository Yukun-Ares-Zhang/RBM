"""
This improved version of RBM use KL divergence and CD method to train itself.
"""

import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import xlwt

U = 4.
T = 0.15
Nx = 8
Ny = Nx
N = Nx * Ny
h_num = 100
DATA_NUM = 50000


class RBM(nn.Module):
   def __init__(self,
               U=4, 
               T=0.15,
               n_vis=64,
               n_hin=100):
        super(RBM, self).__init__()
        self.W = nn.Parameter(torch.randn(n_hin, n_vis)*1e-2)
        self.v_bias = nn.Parameter(torch.zeros(n_vis))
        self.h_bias = nn.Parameter(torch.zeros(n_hin))
        self.U = U
        self.T = T
        self.n_vis = n_vis
        self.n_hin = n_hin
    
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
        
   def forward(self,v):
        pre_h,h = self.v_to_h(v)
        return h
    
   def free_energy(self,v):
        vbias_term = v.mv(self.v_bias)
        wx_b = F.linear(v,self.W,self.h_bias)
        hidden_term = wx_b.exp().add(1).log().sum(1)
        return (- hidden_term - vbias_term)
