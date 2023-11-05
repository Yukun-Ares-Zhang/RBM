import numpy as np
import matplotlib.pyplot as plt

def ising_model_2d(N, J, T, n_steps):
    """
    Monte Carlo simulation of the 2D Ising model using the Metropolis algorithm.
    
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
    lattice = np.ones([N, N])
    
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
    
    markersize = 2
    
    fig = plt.figure()
    ax1 = fig.add_subplot(2,2,1)
    ax1.plot(range(len(M)), M, "ks-", markersize=markersize)
    ax1.set_title("M (Metropolis N="+str(N)+")")
    ax1.set_xlabel("step")
    ax1.set_ylabel("M")
    
    ax2 = fig.add_subplot(2,2,2)
    ax2.plot(range(len(E)), E, "bo-", markersize=markersize)
    ax2.set_title("E (Metropolis N="+str(N)+")")
    ax2.set_xlabel("step")
    ax2.set_ylabel("E")
    
    ax3 = fig.add_subplot(2,2,3)
    ax3.plot(range(len(ChiM)), ChiM, "ks-", markersize=markersize)
    ax3.set_title("ChiM (Metropolis N="+str(N)+")")
    ax3.set_xlabel("step")
    ax3.set_ylabel("ChiM")
    
    ax4 = fig.add_subplot(2,2,4)
    ax4.plot(range(len(ChiE)), ChiE, "bo-", markersize=markersize)
    ax4.set_title("ChiE (Metropolis N="+str(N)+")")
    ax4.set_xlabel("step")
    ax4.set_ylabel("ChiE")
    plt.savefig("../figs&txts/Metropolis/N"+str(N)+"_J"+str(J)+"_T"+str(T)+"_nstep"+str(n_steps)+".png")
    
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
    
    if M_avr<1:
        print(lattice)
    
    return M_avr, E_avr, C_avr, X_avr

# Example usage
print(ising_model_2d(8, 1.0, 0.3, 2000))
