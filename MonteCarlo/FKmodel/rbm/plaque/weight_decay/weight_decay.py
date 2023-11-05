import numpy as np
import time
import os
import matplotlib.pyplot as plt

class RBM():
    def __init__(self, visible_units, hidden_units, learning_rate, weight_decay, batch_size, n_iter):
        self.visible_units = visible_units
        self.hidden_units = hidden_units
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.n_iter = n_iter

        # Initialize weights and biases
        self.weights = np.random.normal(0, 0.01, (visible_units, hidden_units))
        self.visible_bias = np.ones(visible_units) * 0.5
        self.hidden_bias = np.zeros(hidden_units)

    def sigmoid(self, x):
        return np.vectorize(lambda x: 1 - 1 / (1 + np.exp(x)) if x < 0 else 1 / (1 + np.exp(-x)))(x)

    def _fit(self, input_data):
        # Positive phase
        hidden_probabilities = self.sigmoid(np.dot(input_data, self.weights) + self.hidden_bias)
        hidden_states = np.random.binomial(1, hidden_probabilities)

        # Negative phase
        visible_probabilities = self.sigmoid(np.dot(hidden_states, self.weights.T) + self.visible_bias)
        hidden_probabilities = self.sigmoid(np.dot(visible_probabilities, self.weights) + self.hidden_bias)

        # Update weights and biases
        self.weights += self.learning_rate * (np.dot(input_data.T, hidden_probabilities) - np.dot(visible_probabilities.T, hidden_probabilities))
        self.visible_bias += self.learning_rate * np.sum(input_data - visible_probabilities, axis=0)
        self.hidden_bias += self.learning_rate * np.sum(hidden_probabilities, axis=0)

        # Apply L2 regularization
        self.weights -= self.weight_decay * self.weights
        self.visible_bias -= self.weight_decay * self.visible_bias
        self.hidden_bias -= self.weight_decay * self.hidden_bias

    def fit(self, training_data):
        n_samples = training_data.shape[0]
        self.weights = np.random.normal(0, 0.01, (self.visible_units, self.hidden_units))
        self.visible_bias = np.ones(self.visible_units) * 0.5
        self.hidden_bias = np.zeros(self.hidden_units)
        
        # Compute the number of batches
        n_batches = int(np.ceil(training_data.shape[0] / float(self.batch_size)))

        # Now loop over your batches
        for index_iter in range(self.n_iter):
            for batch_idx in range(n_batches):
                start = batch_idx * self.batch_size
                end = start + self.batch_size
                training_data_batch = training_data[start:end, :]
                self._fit(training_data_batch)
                    
    def free_energy(self, v):
        wx_b = np.dot(v, self.weights) + self.hidden_bias
        visible_bias_term = np.dot(v, self.visible_bias)
        hidden_bias_term = np.sum(np.log(1 + np.exp(wx_b)), axis=1)
        return -visible_bias_term - hidden_bias_term

if __name__=='__main__':

    total_reftime = time.time()
    X = np.load("../data_gen/Traning_data.npy")
    Training_v = X[:, 0:-1]
    Training_erg = X[:, -1]
    Y = np.load("../data_gen/Test_data.npy")
    Test_v = Y[:, 0:-1]
    Test_erg = Y[:, -1]
    
    for lr in [ 0.01, 0.001, 0.0001 ]:
        reftime = time.time()
        if os.path.exists("data")==False:
            os.makedirs("data")
        os.chdir("data")
        if os.path.exists("lr="+str(lr))==False:
            os.makedirs("lr="+str(lr))
        os.chdir("lr="+str(lr))
        
        for n_iteration in [10, 100, 1000]: 
            if os.path.exists("n="+str(n_iteration))==False:
                os.makedirs("n="+str(n_iteration))
            os.chdir("n="+str(n_iteration))   
            rbm = RBM(visible_units=64, hidden_units=100, learning_rate=lr, weight_decay=0.1, batch_size=10, n_iter=n_iteration)
            rbm.fit(Training_v)
            
            Predict_erg = -rbm.free_energy(Test_v)
            Test_erg -= Test_erg.min()
            Predict_erg -= Predict_erg.min()
            erg_tuple = []
            for i in range(Test_erg.shape[0]):
                erg_tuple += [(Predict_erg[i], Test_erg[i])]
            erg_tuple = sorted(erg_tuple, key=lambda x:x[1])
            with open("./energy.txt", "w") as f1:
                print("Pred\tExac", file = f1)
                for i in range(Test_erg.shape[0]):
                    print("%5.2f\t%5.2f"%(erg_tuple[i][0], erg_tuple[i][1]),file = f1)
            np.savetxt("weights.txt", rbm.weights, fmt="%5.6f", delimiter="\t")
            np.savetxt("intercept_hidden.txt", rbm.hidden_bias, fmt="%5.6f", delimiter="\t")
            np.savetxt("intercept_visible.txt", rbm.visible_bias, fmt="%5.6f", delimiter="\t")
            os.chdir("..")
        
        erg1=np.loadtxt("n=10/energy.txt", skiprows=1)
        erg2=np.loadtxt("n=100/energy.txt", skiprows=1)
        erg3=np.loadtxt("n=1000/energy.txt", skiprows=1)
        os.chdir("..\\..")
        if os.path.exists("fig")==False:
            os.makedirs("fig")
        os.chdir("fig")
        
        index = np.arange(1, erg1.shape[0]+1, 1)
        fig = plt.figure(figsize=(18, 6))
        fig.tight_layout(h_pad=4)
        ax1 = fig.add_subplot(1,3,1)
        ax1.scatter(index, erg1[:,0], marker="o", c="b")
        ax1.scatter(index, erg1[:,1], marker="o", c="r")
        ax1.set_title("n_iteration=10")
        ax1.set_xlabel("test sample")
        ax1.set_ylabel("erg")

        ax2 = fig.add_subplot(1,3,2)
        ax2.scatter(index, erg2[:,0], marker="o", c="b")
        ax2.scatter(index, erg2[:,1], marker="o", c="r")
        ax2.set_title("n_iteration=100")
        ax2.set_xlabel("test sample")
        ax2.set_ylabel("erg")
        
        ax3 = fig.add_subplot(1,3,3)
        ax3.scatter(index, erg3[:,0], marker="o", c="b")
        ax3.scatter(index, erg3[:,1], marker="o", c="r")
        ax3.set_title("n_iteration=1000")
        ax3.set_xlabel("test sample")
        ax3.set_ylabel("erg")
        fig.suptitle("lr="+str(lr))
        plt.savefig("lr="+str(lr)+".png")
        os.chdir("..")
        
        print("finished lr="+str(lr)+" in %5.2f min, Total time: %5.2f min"%((time.time()-reftime)/60,(time.time()-total_reftime)/60))