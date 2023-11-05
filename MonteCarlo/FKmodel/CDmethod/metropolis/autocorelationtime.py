# test the autocorelation time using M and E
# autocorelation time is tested using integration method or decay criterion 1/e

import numpy as np
import matplotlib.pyplot as plt
import math

def autocorelationtime_ising_model_2d(N, J, T, n_steps):
    """
    calculate the autocorelationtime of Monte Carlo simulation of the 2D Ising model using the Metropolis algorithm.
    
    Parameters
    ----------
    N : int
        The size of the lattice (N x N).
    J : float
        The coupling constant.
    T : float
        The temperature.
    n_steps : int
        The number of Monte Carlo steps to perform.
    
    Returns
    -------
    autocorelationtime
    
    
    """
    # Initialize the lattice
    lattice = np.random.choice([-1, 1], size=(N, N))
    
    # Compute the initial energy
    Energy = 0
    for i in range(N):
        for j in range(N):
            Energy -= J * lattice[i, j] * (lattice[(i+1)%N, j] + lattice[i, (j+1)%N])
    
    M = []
    E = []
    # Perform the Monte Carlo steps
    for step in range(n_steps):
        for itr in range(N**2):
            # Choose a random spin to flip
            i = np.random.randint(N)
            j = np.random.randint(N)
            
            # Compute the energy change
            dEnergy = 2 * J * lattice[i, j] * (lattice[(i+1)%N, j] + lattice[(i-1)%N, j] + lattice[i, (j+1)%N] + lattice[i, (j-1)%N])
            
            # Decide whether to accept the move
            if dEnergy < 0 or np.random.rand() < np.exp(-dEnergy / T):
                lattice[i, j] *= -1
                Energy += dEnergy
        M = np.append(M, [lattice.sum() / N**2])
        E = np.append(E, [Energy / N**2])

    ChiM = []
    ChiE = []
    for t in range(1,int(n_steps/10)):
        ChiM = np.append(ChiM, [np.matmul(M[:(len(M)-t)], M[t:].reshape(-1,1))/(n_steps-t) - M[:(len(M)-t)].sum()/(n_steps-t) * M[t:].sum()/(n_steps-t)])
        ChiE = np.append(ChiE, [np.matmul(E[:(len(E)-t)], E[t:].reshape(-1,1))/(n_steps-t) - E[:(len(M)-t)].sum()/(n_steps-t) * E[t:].sum()/(n_steps-t)])
    
    ChiM = ChiM / ChiM[0]
    ChiE = ChiE / ChiE[0]
    
    tau_M_int = ChiM.sum()
    tau_E_int = ChiE.sum()
    
    # for index in range(len(ChiM)):
    #     if ChiM[index] < 1/math.e:
    #         tau_M_dec = index+1
    #         break
    
    # for index in range(len(ChiE)):
    #     if ChiE[index] < 1/math.e:
    #         tau_E_dec = index+1
    #         break
    
    # markersize = 2
    
    # fig = plt.figure()
    # ax1 = fig.add_subplot(2,2,1)
    # ax1.plot(range(len(M)), M, "ks-", markersize=markersize)
    # ax1.set_title("M (Metropolis N="+str(N)+")")
    # ax1.set_xlabel("step")
    # ax1.set_ylabel("M")
    
    # ax2 = fig.add_subplot(2,2,2)
    # ax2.plot(range(len(E)), E, "bo-", markersize=markersize)
    # ax2.set_title("E (Metropolis N="+str(N)+")")
    # ax2.set_xlabel("step")
    # ax2.set_ylabel("E")
    
    # ax3 = fig.add_subplot(2,2,3)
    # ax3.plot(range(len(ChiM)), ChiM, "ks-", markersize=markersize)
    # ax3.set_title("ChiM (Metropolis N="+str(N)+")")
    # ax3.set_xlabel("step")
    # ax3.set_ylabel("ChiM")
    
    # ax4 = fig.add_subplot(2,2,4)
    # ax4.plot(range(len(ChiE)), ChiE, "bo-", markersize=markersize)
    # ax4.set_title("ChiE (Metropolis N="+str(N)+")")
    # ax4.set_xlabel("step")
    # ax4.set_ylabel("ChiE")
    # plt.savefig("../figs&txts/Metropolis/N"+str(N)+"_J"+str(J)+"_T"+str(T)+"_nstep"+str(n_steps)+".png")
    
    return tau_M_int, tau_E_int

# Example usage
N = 8
J = 1.0
T = 0.1
T_record = []
tau_M_mean = []
tau_M_std = []
tau_E_mean = []
tau_E_std = []
while(T<=5.0):
    tau_M_results = []
    tau_E_results = []
    for test_index in range(10):
        if T > 2.05 and T < 2.35:
            tau_M_int, tau_E_int = autocorelationtime_ising_model_2d(N, J, T, 50000)
        elif T > 2.35 and T < 2.95:
            tau_M_int, tau_E_int = autocorelationtime_ising_model_2d(N, J, T, 10000)
        else:
            tau_M_int, tau_E_int = autocorelationtime_ising_model_2d(N, J, T, 1000)
        
        tau_M_results = np.append(tau_M_results, [tau_M_int])    
        tau_E_results = np.append(tau_E_results, [tau_E_int])
    tau_M_mean = np.append(tau_M_mean, [tau_M_results.mean()])
    #tau_M_std = np.append(tau_M_std, [tau_M_results.std(ddof=1)])
    tau_E_mean = np.append(tau_E_mean, [tau_E_results.mean()])
    #tau_E_std = np.append(tau_E_std, [tau_E_results.std(ddof=1)])
    T_record = np.append(T_record, [T])
    print("T=",T,"finished.")
    T += 0.1       

markersize = 4
fig = plt.figure()
fig.tight_layout(h_pad=4)
ax1 = fig.add_subplot(2,1,1)
ax1.plot(T_record, tau_M_mean, "o-b", markersize=markersize)
# ax1.errorbar(T_record, tau_M_mean, yerr=tau_M_std, fmt="o-b", markersize=markersize, capsize=2)
ax1.set_title("from M")
ax1.set_xlabel("T")
ax1.set_ylabel("tau_M")

ax2 = fig.add_subplot(2,1,2)
ax2.plot(T_record, tau_E_mean, "o-b", markersize=markersize)
# ax2.errorbar(T_record, tau_E_mean, yerr=tau_E_std, fmt="o-b", markersize=markersize, capsize=2)
ax2.set_title("from E")
ax2.set_xlabel("T")
ax2.set_ylabel("tau_E")
fig.suptitle("autocorelation time (Metropolis N="+str(N)+")")

plt.savefig("../figs&txts/autocorelationtime/N"+str(N)+"_J"+str(J)+"autocorelationtime.png")
