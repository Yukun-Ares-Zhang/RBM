'''
'''
from __future__ import print_function
import pyalps
import pyalps.alea as alpsalea # has to use float64 
import numpy as np 
import progressbar
from plot_weights import plot_weights
import sklearn 
import sys  
import os 

class Simulation:
    def __init__(self, lattice, net, rbm, T,batch_size, rng, n_gibbs, n_metropolis, resfolder, update, loadweights=False, J=1):

        self.lattice = lattice
        self.L = int(np.sqrt(self.lattice.Nsite)) # works with square lattice
        self.net = net 
        self.rbm = rbm  #TODO: merge these two
        self.T = T 
        self.batch_size = batch_size
        self.rng = rng 
        self.n_gibbs = n_gibbs
        self.n_metropolis = n_metropolis
        self.resfolder = resfolder 
        self.update = update
        self.J = J  # J=1 FM bond, J=-1 AF bond

        f = resfolder+"/weights.h5"
        if os.path.isfile(f) and loadweights:
            self.net.load_weights(f)
            print ("loaded weights from", f)

        # Init observables
        if update!='wolff':
            self.acceptance = alpsalea.RealObservable('acceptance')
        self.E = alpsalea.RealObservable('E')
        self.E2 = alpsalea.RealObservable('E2')
        self.M = alpsalea.RealObservable('M')
        self.M2 = alpsalea.RealObservable('M2')
        self.M4 = alpsalea.RealObservable('M4')
        self.Mabs = alpsalea.RealObservable('Mabs')
        
        #initial spin configuration and energy 
        self.spins = 2*self.rng.randint(2, size=lattice.Nsite)-1
        self.energy = self.calc_energy(self.spins)
        #self.Emax = 2*self.lattice.Nsite # the maximum energy, assuing square lattice

    def run(self,ntherm,ntrain,nmeasure):

        for n in range(ntherm): 
            self.step(n, phase='therm')
        print ('Thermalization Done')
        
        if self.update =='RBM':
            self.net.X_test, self.net.y_test = self.collect_samples(5000)
            print ('Collect tests Done')
            for n in range(ntrain):
                self.step(n, phase='train')
                #make plot of W 
                if (n%100==0):
                    #plot weights 
                    W, b, a = self.net.get_weights()
                    plot_weights(W, n/100, self.resfolder)

                    #write out tests 
                    y_predict = self.net.model.predict(self.net.X_test, batch_size=1, verbose=1)
                    f = open(self.resfolder+'/fit.dat','w')
                    for i in range(len(self.net.y_test)):
                        f.write('%g %g %g' %(i, self.net.y_test[i], y_predict[i])+'\n')
                    f.close()

            self.net.save_weights(self.resfolder+'/weights.h5')
            print ('Train RBM Done')
    
        for n in range(nmeasure):
            self.step(n, phase='measure')
            self.measure()

    def calc_energy(self, spins):
        '''
        energy of the physical Ising model
        '''
        return 0.5*self.J*np.dot(spins, self.lattice.Kmat.dot(spins)).astype(np.float64)

    def calc_magnetization(self, spins):
        '''
        magnetimzation of the physical Ising model
        '''
        return spins.sum().astype(np.float64)

    def wolff_update(self):
        
        for sw in range(5):
            pc = 1.- np.exp(-2./self.T)
            stck = []
            s = self.rng.randint(self.lattice.Nsite)
            so = self.spins[s]
            self.spins[s] = -so 
            stck.append(s)
            while stck: # if not empty 
                sc = stck.pop() 
                for n in range(self.lattice.Kmat.indptr[sc], self.lattice.Kmat.indptr[sc + 1]):
                    sn = self.lattice.Kmat.indices[n] 
                    if (self.spins[sn]==so and self.rng.rand()< pc):
                        stck.append(sn)
                        self.spins[sn] = -so 

        #update the energy 
        self.energy = self.calc_energy(self.spins)
    
    def metropolis_update(self):
        '''
        a sweep through the lattice 
        '''

        for sw in range(self.lattice.Nsite):
            i = self.rng.randint(self.lattice.Nsite)
            spinsnew = self.spins.copy()
            spinsnew[i] = -spinsnew[i]
            
            energy = self.calc_energy(spinsnew)
            
            ratio = np.exp(-(energy-self.energy)/self.T) 
            
            if (ratio > rng.rand()):
                self.acceptance << 1.0 
                self.spins = spinsnew 
                self.energy = energy
            else:
                self.acceptance << 0.0
    
    def collect_samples(self, n):
        '''
        collect samples 
        '''
        X = []
        y = []
        for i in range(n):
            self.metropolis_update() 
            #data argumentation
            #x = (self.spins+1)/2
            #for j in range(5): 
            #    x = self.random_shift(x)
            #    X.append(x)
            #    y.append(-self.energy/self.T)

            X.append((self.spins+1)/2)
            y.append(-self.energy/self.T)

        X = np.array(X)
        y = np.array(y)
        X, y = sklearn.utils.shuffle(X, y)

        return X, y 

    def train_rbm(self, verbose):
        '''
        collect samples of batch_size 
        then train the rbm 
        '''

        X , y = self.collect_samples(self.batch_size)
        self.net.fit(X, y, verbose)

        W, b, a = self.net.get_weights()
        #print (W.shape)
        #print (b.shape)
        
        self.rbm.set_weights(W, b, a)

    def step(self, n, phase='therm'):
        '''
        '''

        if phase=='therm':
            self.metropolis_update() 
        elif phase=='train':
            self.metropolis_update() 
            verbose = True if (n%100==0) else False
            self.train_rbm(verbose) 
        else: 
            if self.update == 'RBM':
               self.rbm_update()
            elif self.update == 'wolff':
               self.wolff_update() 
            elif self.update == 'metropolis':
               self.metropolis_update() 
            else:
                print ('what ?')
                sys.exit(1)

    def rbm_update(self):

        spinsnew = self.spins.copy()
        for gibbs in range(self.n_gibbs):
            vold = (spinsnew+1)/2
            vold.shape = (1, -1)
        
            hold = self.rbm._sample_hiddens(vold, self.rng)
            hold.shape = (1, -1)
            
            for metro in range(self.n_metropolis):
                # sample a hnew with metropolis 
                i = self.rng.randint(self.rbm.n_components)
                hnew = hold.copy()
                hnew[0, i] = 1-hnew[0, i] 
             
                ratio = np.exp(-(self.rbm._free_energy_hidden(hnew) - self.rbm._free_energy_hidden(hold)))
                if (ratio > rng.rand()):
                    hold = hnew 
        
            # sample a new v 
            vnew = self.rbm._sample_visibles(hold, self.rng)
            spinsnew = 2*vnew -1 
            spinsnew.shape = (self.lattice.Nsite,)
            
        #translation shift of the spin configuration 
        #spinsnew = self.random_shift(spinsnew)

        #accept or not 
        #energy of the physical model 
        energy = self.calc_energy(spinsnew)
        
        ratio1 = np.exp(self.rbm._free_energy(vnew) - self.rbm._free_energy(vold))[0]
        ratio2 = np.exp(-(energy-self.energy)/self.T) 

        nflip = np.count_nonzero(self.spins!=spinsnew)
        #print (ratio1, ratio2 , ratio1 * ratio2, nflip) 

        if (ratio1*ratio2 > self.rng.rand()):
            self.acceptance << 1.0 
            self.spins = spinsnew 
            self.energy = energy 
        else:
            self.acceptance << 0.0

    def random_shift(self, spins):
        '''
        random shift the configuration 
        '''
        X = spins.copy() 
        X.shape = (self.L, self.L)
        X = np.roll(X, self.rng.randint(self.L), axis=0)
        X = np.roll(X, self.rng.randint(self.L), axis=1)

        return X.reshape(self.lattice.Nsite,)

    def measure(self):

        e = self.energy/self.lattice.Nsite
        m = self.calc_magnetization(self.spins)/self.lattice.Nsite 

        # Add sample to observables
        self.E << e
        self.E2 << e**2

        self.M << m
        self.M2 << m**2
        self.M4 << m**4 
        self.Mabs << abs(m)

if __name__ == '__main__': 
    from lattice import Lattice 
    from network import Network
    #from cnn import Network
    from rbm import BernoulliRBM
    import argparse
    import re

    parser = argparse.ArgumentParser(description='Perform MC calculation')
    parser.add_argument("-latticename", default='square lattice',  help="lattice structure")
    parser.add_argument('-L', type=int, default=8)
    parser.add_argument('-T', type=float, default=2.3)
    parser.add_argument('-J', type=float, default=1.0)

    parser.add_argument('-batch_size', type=int, default=10)
    parser.add_argument('-n_hidden', type=int, default=100)
    parser.add_argument('-n_metropolis', type=int, default=0)
    parser.add_argument('-n_gibbs', type=int, default=1)

    parser.add_argument("-ntherm", type=int, default=10000, help="ntherm")
    parser.add_argument("-ntrain", type=int, default=10000, help="ntrain")
    parser.add_argument("-nmeasure", type=int, default=10000, help="nmeasure")
    parser.add_argument("-folder",default='../../data/online/' ,help="folder")

    parser.add_argument('-update', default='metropolis', help='update methods')
    parser.add_argument("-loadweights", action='store_true',  help="")

    args = parser.parse_args()

    resfolder = args.folder + '/'  \
               + args.latticename.replace(" ", '') + '_' \
               + 'L' + str(args.L) + '_' \
               + 'T' + str(args.T) + '_' \
               + 'J' + str(args.J) + '_' \
               + 'nhidden' + str(args.n_hidden) + '_'
    
    if not os.path.isdir(resfolder):
        os.makedirs(resfolder)

    from numpy.random.mtrand import RandomState
    rng = RandomState() # no seed here since we want to have different configs for different run 
   
    lattice = Lattice(args.latticename, args.L, args.L)

    net = Network(lattice.Nsite, args.n_hidden)
    rbm = BernoulliRBM(n_components=args.n_hidden)
 
    sim = Simulation(lattice, net, rbm, args.T, args.batch_size, rng, n_gibbs=args.n_gibbs, n_metropolis=args.n_metropolis,resfolder=resfolder, update=args.update, loadweights=args.loadweights, J=args.J) 
    sim.run(args.ntherm, args.ntrain, args.nmeasure)
    
    if args.update!='wolff':
        print ('acc', sim.acceptance.mean, '+/-', sim.acceptance.error)
    print ('E', sim.E.mean,'+/-', sim.E.error, sim.E.tau)
    print ('E2', sim.E2.mean,'+/-', sim.E2.error, sim.E2.tau)

    print ('M', sim.M.mean,'+/-', sim.M.error, sim.M.tau )
    print ('M2', sim.M2.mean,'+/-', sim.M2.error, sim.M2.tau )
    print ('M4', sim.M4.mean,'+/-', sim.M4.error, sim.M4.tau )
    print ('|M|', sim.Mabs.mean,'+/-', sim.Mabs.error, sim.Mabs.tau )

    m4 = alpsalea.MCScalarData(sim.M4.mean, sim.M4.error)
    m2 = alpsalea.MCScalarData(sim.M2.mean, sim.M2.error)

    print ('Binder', m4/(m2*m2))
