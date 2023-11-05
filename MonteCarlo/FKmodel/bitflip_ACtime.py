import numpy as np 
import math
import random
import cmath
import copy
import matplotlib.pyplot as plt
from IPython.display import Latex

U = 4.
T = 0.15
N = 64
SWEEP_NUM = 500
UPDATE_TOTAL = SWEEP_NUM * N

def alpha_to_xy(alpha: int):
    if alpha < 0 or alpha >= N:
        return -1, -1
    else:
        x = alpha % int(math.sqrt(N)) + 1
        y = (alpha - x + 1) / int(math.sqrt(N)) + 1
        return x, y

H = np.zeros([N, N])
for i in range(N):
    x, y = alpha_to_xy(i)
    check_list = [i-int(math.sqrt(N)), i-1, i+1, i+int(math.sqrt(N))]
    for j in check_list:
        x1, y1 = alpha_to_xy(j)
        if x1 < 0 or y1 < 0:
            continue
        elif abs(x-x1) > 1 or abs(y-y1) > 1:
            continue
        else:
            H[i, j] = -1 / T

def minus_F_FK(vx: np.ndarray):
    assert vx.ndim == 1 and vx.size == N, \
    "vx is not a valid vector."

    for i in range(N):
        H[i,i] = U / T * (vx[i] - 1/2)
        
    w, B = np.linalg.eig(H)
    minus_F = U / (2 * T) * np.sum(vx) 
    for i in range(N):
        minus_F += cmath.log(1 + cmath.exp(-w[i])).real
    return minus_F


#x = np.random.randint(0,2,N)
#x = np.zeros(N)
x = np.ones(N)
F0 = minus_F_FK(x)

yF_list = [F0]
ym_list = [np.sum(x) / N]
update_num = 0
proposal_num = 0

while update_num < UPDATE_TOTAL:
    flip_i = random.randint(0, N-1)
    x_update = x.copy()
    x_update[flip_i] = 1 - x_update[flip_i]

    F1 = minus_F_FK(x_update)
    #print(F0, F1)
    Accept_rate = min(math.exp(F1 - F0).real, 1) 
    random_n = random.random()
    if(random_n < Accept_rate):
        x = x_update.copy()
        update_num += 1
        F0 = minus_F_FK(x)
        if update_num % 1 == 0:
            yF_list += [F0]
            ym_list += [np.sum(x) / N]
    proposal_num += 1

print("Simulation done.")

print("T/t = ", T, "Acceptance ratio = ", update_num / proposal_num)

plot_yF = np.array(yF_list)
plot_ym = np.array(ym_list)
plot_x = np.linspace(0, np.size(plot_yF) - 1, np.size(plot_yF))
fig = plt.figure()
ax = fig.add_subplot()
ax.plot(plot_x, plot_yF, label="Free Energy", color="blue")

ax2 = ax.twinx()
ax2.plot(plot_x, plot_ym, label="position density", color="red")
ax.set_xlabel("time")
plt.savefig("./plot.png")

t_max = len(yF_list) - 1
ACT_F = []
ACT_F0 = 1
for t in range(int(t_max / 50)):
    ACT_F += [0.]
    for i in range(t_max - t):
        ACT_F[t] += yF_list[i] * yF_list[i + t]
    ACT_F[t] /= (t_max - t)
    ACT_F[t] -= 1 / (t_max - t)**2 * sum(yF_list[:(t_max - t)]) * sum(yF_list[(t_max - t):])
    if t == 0:
        ACT_F0 = ACT_F[0]
    ACT_F[t] /= ACT_F0

plot_x = np.linspace(0, len(ACT_F) - 1, len(ACT_F))

plt.figure()
plt.plot(plot_x, ACT_F, label="Free Energy", color="blue")
plt.xlabel('time')
plt.ylabel('Autocorrelation time of energy')
plt.savefig('./correlation_energy.png')

t_max = len(ym_list) - 1
ACT_m = []
ACT_m0 = 1
for t in range(int(t_max / 50)):
    ACT_m += [0.]
    for i in range(t_max - t):
        ACT_m[t] += ym_list[i] * ym_list[i + t]
    ACT_m[t] /= (t_max - t)
    ACT_m[t] -= 1 / (t_max - t)**2 * sum(ym_list[:(t_max - t)]) * sum(ym_list[(t_max - t):])
    if t == 0:
        ACT_m0 = ACT_m[0]
    ACT_m[t] /= ACT_m0

plot_x = np.linspace(0, len(ACT_m) - 1, len(ACT_m))

plt.figure()
plt.plot(plot_x, ACT_m, label="occupancy density", color="blue")
plt.xlabel('time')
plt.ylabel('Autocorrelation time of occupancy')
plt.savefig('./correlation_occupancy.png')