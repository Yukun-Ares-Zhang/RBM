import numpy as np 
import math
import random
import cmath
import matplotlib.pyplot as plt

U = 4.
T = 0.15
Nx = 8
Ny = Nx
N = Nx * Ny

def alpha_to_xy(alpha: int):
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
    if alpha < 0 or alpha >= N:
        return -1, -1
    else:
        x = int(alpha % Nx)
        y = int(alpha / Nx)
        return x, y
 
class H:
    def __init__(self, U, T, x0: np.ndarray):
        self.hamitonian = np.zeros([N, N])
        for i in range(N):
            x, y = alpha_to_xy(i)
            check_list = [i-Nx, i-1, i+1, i+Nx]
            for j in check_list:
                x1, y1 = alpha_to_xy(j)
                if abs(x-x1)*abs(y-y1) == 0 and abs(x-x1)+abs(y-y1) == 1:
                    self.hamitonian[i, j] = -1 / T
        self.U = U
        self.T = T
        self.x = x0
    
    def update_h(self):
        """ Update the hamitonian with new population vector. """
        for i in range(N):
            self.hamitonian[i, i] = self.U / self.T * (self.x[i] - 1/2)

    def free_erg(self):
        """ Calculate the free energy. """
        H.update_h(self)
        w, B = np.linalg.eig(self.hamitonian)
        F = -self.U / (2 * self.T) * self.x.sum()
        for i in range(N):
            F -= cmath.log(1 + cmath.exp(-w[i]))
        if F.imag / F.real < 1e-7:
            F = F.real
        return F        

    def update(self):
        temp = self.x.copy()#backup
        F0 = H.free_erg(self)
        updated = False
        while updated == False:    
            flip_index = random.randint(0, N-1)
            self.x[flip_index] = 1 - self.x[flip_index]
            F1 = H.free_erg(self)
            accept_rate = min(math.exp(F0 - F1), 1)
            rand = random.random()
            if rand > accept_rate:
                self.x = temp.copy()
            else:
                updated = True

        #print("{}{:>10.2f}{:>10.2f}{:>10.2f}{:>10.2f}".format(updated, F0, F1, accept_rate, rand))
        
    
x0 = np.ones(N, dtype = int)
h = H(U, T, x0)
plot_F = []
plot_m = []
for i in range(1000):
    h.update()
    plot_F += [h.free_erg()]
    plot_m += [h.x.sum() / N]
plot_x = list(range(len(plot_F)))
fig = plt.figure()
ax = fig.add_subplot()
ax.plot(plot_x, plot_F, label="Free Energy", color="blue")
ax2 = ax.twinx()
ax2.plot(plot_x, plot_m, label="position density", color="red")
ax.set_xlabel("time")
plt.savefig("./bitflip_classPLOT/F&m_plot.png")