'''
'''
from __future__ import print_function
import pyalps
import pyalps.alea as alpsalea # has to use float64 
import numpy as np 
import progressbar

class Simulation:
    def __init__(self, lattice, T, rbm, rng, n_gibbs, n_metropolis, update):

        self.lattice = lattice
        self.T = T 
        self.rbm = rbm 
        self.rng = rng 
        self.n_gibbs = n_gibbs
        self.n_metropolis = n_metropolis
        self.update = update

        # Init observables
        self.E = alpsalea.RealObservable('E')
        self.E2 = alpsalea.RealObservable('E2')
        self.M = alpsalea.RealObservable('M')
        self.M2 = alpsalea.RealObservable('M2')
        self.M4 = alpsalea.RealObservable('M4')
        self.Mabs = alpsalea.RealObservable('Mabs')
        if update!='wolff':
            self.acceptance = alpsalea.RealObservable('acceptance')
            self.flipratio = alpsalea.RealObservable('flipratio')
        if update=='RBM' and n_metropolis>0 :
            self.acceptance2 = alpsalea.RealObservable('acceptance2')
        
        #initial spin configuration and energy 
        self.spins = 2*self.rng.randint(2, size=lattice.Nsite)-1
        self.energy = self.calc_energy(self.spins)

    def run(self,ntherm,nsweep,nskip):
        # Thermalize for ntherm steps

        print ('Thermalization:')
        bar = progressbar.ProgressBar(maxval=ntherm, \
                    widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()]).start()

        for n in range(ntherm): 
            #print 'thermalization, n',n
            self.step()
            bar.update(n+1)
        bar.finish()

        # Run nsweep steps
        print ('Measurement:')
        bar = progressbar.ProgressBar(maxval=nsweep, \
                    widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()]).start()
        for n in range(nsweep):
            #print 'sweep,n', n
            for i in range(nskip): self.step()
            self.measure()
            bar.update(n+1)
        bar.finish()

    def calc_energy(self, spins):
        '''
        energy of the physical Ising model
        '''
        return 0.5*np.dot(spins, self.lattice.Kmat.dot(spins)).astype(np.float64)

    def calc_magnetization(self, spins):
        '''
        magnetimzation of the physical Ising model
        '''
        return spins.sum().astype(np.float64)

    def step(self):
        '''
        '''
        #for sweep in range(self.lattice.Nsite):
        for sweep in range(1):

            if self.update=='RBM':
            
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
                            self.acceptance2 << 1.0 
                        else:
                            self.acceptance2 << 0.0 
             
                    # sample a new v 
                    vnew = self.rbm._sample_visibles(hold, self.rng)
                    spinsnew = 2*vnew -1 
                    spinsnew.shape = (self.lattice.Nsite,)
             
                # accept or not 
                #energy of the physical model 
                energy = self.calc_energy(spinsnew)
                
                ratio1 = np.exp(self.rbm._free_energy(vnew) - self.rbm._free_energy(vold))
                ratio2 = np.exp(-(energy-self.energy)/self.T) 
                #print (ratio1, ratio2)
             
                if (ratio1* ratio2 > rng.rand()):
                    self.acceptance << 1.0 
                    self.flipratio << float(np.count_nonzero(self.spins!=spinsnew))/self.lattice.Nsite
                    self.spins = spinsnew 
                    self.energy = energy 
                else:
                    self.acceptance << 0.0
            
            elif (self.update=='metropolis'):
            
                i = self.rng.randint(self.lattice.Nsite)
                spinsnew = self.spins.copy()
                spinsnew[i] = -spinsnew[i]
             
                energy = self.calc_energy(spinsnew)
             
                ratio = np.exp(-(energy-self.energy)/self.T) 
             
                if (ratio > rng.rand()):
                    self.acceptance << 1.0 
                    self.flipratio << float(np.count_nonzero(self.spins!=spinsnew))/self.lattice.Nsite
                    self.spins = spinsnew 
                    self.energy = energy
                else:
                    self.acceptance << 0.0
                    self.flipratio << 0.0 
            
            elif (self.update == 'wolff'):
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
            else:
                print ('which way to update ?')
                sys.exit(1)

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
    #from sklearn.neural_network import BernoulliRBM 
    from rbm import BernoulliRBM
    from load_data import load_data 
    import argparse
    import os 
    import re

    parser = argparse.ArgumentParser(description='Perform MC calculation')
    parser.add_argument("-latticename", default='square lattice',  help="lattice structure")
    parser.add_argument('-filename', default='../spins/ising_L8_T2.2691853.dat')
    parser.add_argument('-modelname', default=None)

    parser.add_argument('-update', default='metropolis', help='update methods')

    parser.add_argument('-n_hidden', type=int, default=64)
    parser.add_argument('-n_iter', type=int, default=50)
    parser.add_argument('-n_metropolis', type=int, default=1)
    parser.add_argument('-n_gibbs', type=int, default=1)

    parser.add_argument("-ntherm", type=int, default=1000, help="ntherm")
    parser.add_argument("-nmeasure", type=int, default=10000, help="nmeasure")
    parser.add_argument("-nskip", type=int, default=1, help="nskip")
    parser.add_argument("-folder",default='../../data/mc/' ,help="folder")

    args = parser.parse_args()

    from numpy.random.mtrand import RandomState
    rng = RandomState(42)

    L = int(re.search('L([0-9]*)_',args.filename).group(1)) 
    T = float(re.search('T(-?[0-9]*\.?[0-9]*)_',args.filename).group(1)) 
    
    to_fit = True 
    if 'nhidden' in args.filename:
        args.n_hidden = int(re.search('nhidden([0-9]*)_',args.filename).group(1)) 
        to_fit = False 
        print ('use {} hidden variables'.format(args.n_hidden))
    else:
        X_train, y_train, X_test, y_test = load_data(args.filename) 
        print ('load training data done')

    resfolder = os.path.dirname(args.filename)
   
    if not os.path.isdir(resfolder):
        os.makedirs(resfolder)

    lattice = Lattice(args.latticename, L, L)

    rbm = BernoulliRBM(n_components=args.n_hidden, n_iter=args.n_iter, batch_size=50, learning_rate=0.01,verbose=1, random_state=rng, resfolder=resfolder)
    if args.update=='RBM':
        if to_fit:
            rbm.fit(X_train)
            print ('fit rbm done')
        else:
            rbm.load(args.filename, lattice.Nsite)
            print ('load rbm done')


    sim = Simulation(lattice, T, rbm, rng, n_gibbs=args.n_gibbs, n_metropolis=args.n_metropolis, update = args.update)
    sim.run(args.ntherm, args.nmeasure, args.nskip)

    if args.update!='wolff':
        print ('acc', sim.acceptance.mean, '+/-', sim.acceptance.error)
        print ('flipratio', sim.flipratio.mean, '+/-', sim.flipratio.error)
    if args.update=='RBM' and args.n_metropolis>0 :
        print ('acc2', sim.acceptance2.mean,'+/-', sim.acceptance2.error)
    
    print ('E', sim.E.mean,'+/-', sim.E.error, sim.E.tau)
    print ('E2', sim.E2.mean,'+/-', sim.E2.error, sim.E2.tau )

    print ('M', sim.M.mean,'+/-', sim.M.error, sim.M.tau )
    print ('M2', sim.M2.mean,'+/-', sim.M2.error, sim.M2.tau )
    print ('M4', sim.M4.mean,'+/-', sim.M4.error, sim.M4.tau )
    print ('|M|', sim.Mabs.mean,'+/-', sim.Mabs.error, sim.Mabs.tau )

    m4 = alpsalea.MCScalarData(sim.M4.mean, sim.M4.error)
    m2 = alpsalea.MCScalarData(sim.M2.mean, sim.M2.error)

    print ('Binder', m4/(m2*m2))

    #print 'ratio1', sim.ratio1.mean, sim.ratio1.error 
    #print 'ratio2', sim.ratio2.mean, sim.ratio2.error 
