import numpy as np
import matplotlib.pyplot as plt
import os
import random

class ising_model_2d:
    def __init__(self, N:int, h:float, J1:float, J2:float, K:float, T:float):
        self.N = N
        self.h = h
        self.J1 = J1
        self.J2 = J2
        self.K = K
        self.T = T
        self.lattice = np.random.choice([1,-1],(N, N))
        # Compute the initial energy
        Energy = 0
        for i in range(N):
            for j in range(N):
                Energy -= h * self.lattice[i, j]
                Energy -= J1 * self.lattice[i, j] * (self.lattice[(i+1)%N, j] + self.lattice[i, (j+1)%N])
                Energy -= J2 * self.lattice[i, j] * (self.lattice[(i+1)%N, (j+1)%N] + self.lattice[(i-1)%N, (j+1)%N])
                Energy -= K * self.lattice[i, j] * self.lattice[(i+1)%N, j] * self.lattice[i, (j+1)%N] * self.lattice[(i+1)%N, (j+1)%N] 
        self.Energy = Energy
    
    def update(self):
        """
        update in Metropolis
        """
        # Choose a random spin to flip
        i = np.random.randint(N)
        j = np.random.randint(N)
        
        # Compute the energy change
        dEnergy = 2 * h * self.lattice[i, j]
        dEnergy += 2 * J1 * self.lattice[i, j] * (self.lattice[(i+1)%N, j] + self.lattice[(i-1)%N, j] + self.lattice[i, (j+1)%N] + self.lattice[i, (j-1)%N])
        dEnergy += 2 * J2 * self.lattice[i, j] * (self.lattice[(i+1)%N, (j+1)%N] + self.lattice[(i+1)%N, (j-1)%N] + self.lattice[(i-1)%N, (j+1)%N] + self.lattice[(i-1)%N, (j-1)%N])
        dEnergy += 2 * K * self.lattice[i, j] * \
            (self.lattice[(i+1)%N, j] * self.lattice[i, (j+1)%N] * self.lattice[(i+1)%N, (j+1)%N] + \
            self.lattice[(i-1)%N, j] * self.lattice[i, (j+1)%N] * self.lattice[(i-1)%N, (j+1)%N] + \
            self.lattice[(i+1)%N, j] * self.lattice[i, (j-1)%N] * self.lattice[(i+1)%N, (j-1)%N] + \
            self.lattice[(i-1)%N, j] * self.lattice[i, (j-1)%N] * self.lattice[(i-1)%N, (j-1)%N])
        
        # Decide whether to accept the move
        if dEnergy < 0 or np.random.rand() < np.exp(-dEnergy / T):
            self.lattice[i, j] *= -1
            self.Energy += dEnergy


if __name__ == "__main__":
    # Example usage
    T = 10.0
    N = 8
    h = 1.0
    J1 = 1.0
    J2 = 0.1
    K = 1.0
    sample_num = 50000
    sample_itv = 5
    
    ising = ising_model_2d(N, h, J1, J2, K, T)
    vectors = []
    ergs = []#negative total energy
    count = 0
    for i in range(100 + sample_itv * sample_num):
        ising.update()
        if (i>=100 and (i-100)%sample_itv == 0):
            if i == 100:
                vectors = ising.lattice.reshape(1, -1)
                ergs = -np.array([ising.Energy])
            else:
                vectors = np.append(vectors, ising.lattice.reshape(1, -1), axis = 0)
                ergs = np.append(ergs, -ising.Energy)
            count += 1
            if count % int(sample_num/5) == 0:
                print(f"Updating... [{count:>10d}/{sample_num}]")
    print("Updated.")
    
    training_data = []
    test_data = []
    sample = random.sample(range(len(ergs)), int(len(ergs) / 5))
    for i in range(len(ergs)):
        if i in sample:
            test_data += [np.append(vectors[i],ergs[i])]    
        else:
            training_data += [np.append(vectors[i], ergs[i])]
            
    test_data = np.array(test_data)
    training_data = np.array(training_data)
    np.save("Traning_data", training_data)
    np.save("Test_data", test_data)
    
    test_erg = test_data[:, -1]
    print(test_erg.shape)
    test_erg = sorted(test_erg)
    fig = plt.figure()
    ax1 = fig.add_subplot(1,1,1)
    ax1.scatter(range(len(test_erg)), test_erg, marker="o", c="b")
    ax1.set_title("test erg")
    ax1.set_xlabel("test sample")
    ax1.set_ylabel("erg")
    plt.savefig("test_erg.png")