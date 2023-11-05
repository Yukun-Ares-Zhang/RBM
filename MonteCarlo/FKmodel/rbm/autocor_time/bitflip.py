import numpy as np
import math
import cmath
import random

 
class H:
    def __init__(self, U, T, Nx: int, Ny: int):
        self.Nx = Nx
        self.Ny = Ny
        self.N = Nx * Ny
        self.hamitonian = np.zeros([self.N, self.N])
        for i in range(self.N):
            x, y = self.alpha_to_xy(i)
            check_list = [i-self.Nx, i-1, i+1, i+self.Nx]
            for j in check_list:
                x1, y1 = self.alpha_to_xy(j)
                if abs(x-x1)*abs(y-y1) == 0 and abs(x-x1)+abs(y-y1) == 1:
                    self.hamitonian[i, j] = -1 / T
        self.U = U
        self.T = T
        self.x = np.ones(self.N, dtype = int)
        
    def alpha_to_xy(self, alpha: int):
        """ convert index: 1D to 2D
        args:
            alpha: 1D index
            index rule sample:
            0 1 2
            3 4 5
            6 7 8
        return:
            x: horizental index
            y: verticle index
        """
        if alpha < 0 or alpha >= self.N:
            return -1, -1
        else:
            x = int(alpha % self.Nx)
            y = int(alpha / self.Nx)
            return x, y
    
    def update_h(self):
        """ Update the hamitonian with new population vector. """
        for i in range(self.N):
            self.hamitonian[i, i] = self.U / self.T * (self.x[i] - 1/2)
            
    def get_eigen_v(self):
        w, B = np.linalg.eig(self.hamitonian)
        return w

    def free_erg(self):
        """ Calculate the free energy. """
        H.update_h(self)
        w, B = np.linalg.eig(self.hamitonian)
        F = -self.U / (2 * self.T) * self.x.sum()
        for i in range(self.N):
            F -= cmath.log(1 + cmath.exp(-w[i]))
        if F.imag / F.real < 1e-7:
            F = F.real
        return F        

    def update(self):
        temp = self.x.copy()#backup
        F0 = H.free_erg(self)
        
        flip_index = random.randint(0, self.N-1)
        self.x[flip_index] = 1 - self.x[flip_index]
        F1 = H.free_erg(self)
        accept_rate = min(math.exp(F0 - F1), 1)
        rand = random.random()
        if rand > accept_rate:
            self.x = temp.copy()
        else:
            self.update_h()
