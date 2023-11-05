import numpy as np
import math
import cmath
import random
import xlwt
import bitflip
import matplotlib.pyplot as plt
import os

U = 4.
T = 0.15
Nx = 8
Ny = Nx
N = Nx * Ny
h_num = 100
DATA_NUM = 10000


x0 = np.ones(N, dtype = int)
h = bitflip.H(U, T, Nx, Ny)
ergs = []#negative free energy
for i in range(200+DATA_NUM):
    h.update()
    if i>=200:
        ergs = np.append(ergs, [-h.free_erg()]) 
        if (i-200)% int(DATA_NUM / 5) == 0:
            print(f"Updating... [{i-200:>10d}/{DATA_NUM}]")
print("Updated.")

ergs -= ergs.min()

ChiE = []
for t in range(1,int(DATA_NUM/10)):
    ChiE = np.append(ChiE, [np.matmul(ergs[:(len(ergs)-t)], ergs[t:].reshape(-1,1))/(len(ergs)-t) - ergs[:(len(ergs)-t)].sum()/(len(ergs)-t) * ergs[t:].sum()/(len(ergs)-t)])

if os.path.exists("fig")==False:
    os.makedirs("fig")
os.chdir("fig")
markersize = 2

fig = plt.figure(figsize=(6, 8))
ax1 = fig.add_subplot(2,1,1)
ax1.plot(range(len(ergs)), ergs, "ks-", markersize=markersize)
ax1.set_title("ergs (Metropolis "+str(N)+"*"+str(N)+")")
ax1.set_xlabel("step")
ax1.set_ylabel("ergs")

ax2 = fig.add_subplot(2,1,2)
ax2.plot(range(len(ChiE)), ChiE, "ks-", markersize=markersize)
ax2.set_title("ChiE (Metropolis "+str(N)+"*"+str(N)+")")
ax2.set_xlabel("step")
ax2.set_ylabel("ChiE")

plt.savefig("ChiE_U"+str(U)+"_T"+str(T)+"_num"+str(DATA_NUM)+".png")
os.chdir("..")
