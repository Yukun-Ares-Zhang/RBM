import numpy as np
import matplotlib.pyplot as plt

def EMCX_ising_model_2d(N, J, T, n_steps):
    """
    Monte Carlo simulation of calculating the EMCX of 2D Ising model using the Metropolis algorithm.
    
    E_avr: energy per site
    M_avr: magnetic moment per site
    C_avr: specific heat per site
    X_avr: magnetic suseptibility
    
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
    lattice : ndarray
        The final configuration of the lattice.
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

    start_index = int(n_steps/10)
    M_eq = M[start_index:]
    E_eq = E[start_index:]
    
    M1 = 0
    M2 = 0
    E1 = 0
    E2 = 0
    index = 0
    sampling_step = int(n_steps/200)
    count = 0
    while(index<len(M_eq)):
        M1+=M_eq[index]
        M2+=M_eq[index]**2
        E1+=E_eq[index]
        E2+=E_eq[index]**2
        count += 1
        index += sampling_step
    
    M1 /= count
    M2 /= count
    E1 /= count
    E2 /= count
    
    M_avr = abs(M1)
    E_avr = E1
    C_avr = N**2/T**2 * (E2 - E1**2)
    X_avr = N**2/T * (M2 - M1**2)
    
    return M_avr, E_avr, C_avr, X_avr

N = 8
J = 1.0
T = 0.1
T_record = []
M_results = []
E_results = []
C_results = []
X_results = []

while(T<=5.0):
    if T > 2.05 and T < 2.35:
        M_avr, E_avr, C_avr, X_avr = EMCX_ising_model_2d(N, J, T, 100000)
    elif T > 2.35 and T < 2.65:
        M_avr, E_avr, C_avr, X_avr = EMCX_ising_model_2d(N, J, T, 20000)
    else:
        M_avr, E_avr, C_avr, X_avr = EMCX_ising_model_2d(N, J, T, 2000)
    M_results = np.append(M_results, [M_avr])
    E_results = np.append(E_results, [E_avr])
    C_results = np.append(C_results, [C_avr])
    X_results = np.append(X_results, [X_avr])
    T_record = np.append(T_record, [T])
    print("T=",T,"finished.")
    T += 0.1

f = open("../figs&txts/MECX/N"+str(N)+"_J"+str(J)+"_MECX.txt", "w")
print("T\tM\tE\tC\tX", file=f)
for i in range(len(T_record)):
    print(T_record[i],"\t",M_results[i],"\t",E_results[i],"\t",C_results[i],"\t",X_results[i], file=f)
f.close()

markersize = 4
fig = plt.figure()
fig.tight_layout(h_pad=4)
ax1 = fig.add_subplot(2,2,1)
ax1.plot(T_record, M_results, "o-b", markersize=markersize)
ax1.set_title("M-T")
ax1.set_xlabel("T")
ax1.set_ylabel("M")

ax2 = fig.add_subplot(2,2,2)
ax2.plot(T_record, E_results, "o-b", markersize=markersize)
ax2.set_title("E-T")
ax2.set_xlabel("T")
ax2.set_ylabel("E")

ax3 = fig.add_subplot(2,2,3)
ax3.plot(T_record, C_results, "o-b", markersize=markersize)
ax3.set_title("C-T")
ax3.set_xlabel("T")
ax3.set_ylabel("C")

ax4 = fig.add_subplot(2,2,4)
ax4.plot(T_record, X_results, "o-b", markersize=markersize)
ax4.set_title("X-T")
ax4.set_xlabel("T")
ax4.set_ylabel("X")
fig.suptitle("MECX-T (Metropolis N="+str(N)+")")

plt.savefig("../figs&txts/MECX/N"+str(N)+"_J"+str(J)+"_MECX.png")