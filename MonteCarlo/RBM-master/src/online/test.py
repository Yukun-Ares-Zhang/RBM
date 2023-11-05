
if __name__=='__main__':
    from lattice import Lattice 
    #from network import Network
    from cnn import Network
    from rbm import BernoulliRBM
    from mc import Simulation
    import argparse
    import os 
    import re

    parser = argparse.ArgumentParser(description='Perform MC calculation')
    parser.add_argument("-latticename", default='square lattice',  help="lattice structure")
    parser.add_argument('-L', type=int, default=8)
    parser.add_argument('-T', type=float, default=2.3)

    parser.add_argument('-n_hidden', type=int, default=100)
    parser.add_argument('-n_metropolis', type=int, default=0)
    parser.add_argument('-n_gibbs', type=int, default=1)

    parser.add_argument("-ntherm", type=int, default=10000, help="ntherm")
    parser.add_argument("-ntrain", type=int, default=100000, help="ntrain")
    parser.add_argument("-nmeasure", type=int, default=100000, help="nmeasure")
    parser.add_argument("-folder",default='../../data/online/' ,help="folder")

    parser.add_argument('-update', default='metropolis', help='update methods')
    parser.add_argument("-loadweights", action='store_true',  help="")

    args = parser.parse_args()

    from numpy.random.mtrand import RandomState
    rng = RandomState() # no seed here since we want to have different configs for different run 
   
    lattice = Lattice(args.latticename, args.L, args.L)

    net = Network(lattice.Nsite, args.n_hidden)
    rbm = BernoulliRBM(n_components=args.n_hidden)
 
    sim = Simulation(lattice, net, rbm, args.T, rng, n_gibbs=args.n_gibbs, n_metropolis=args.n_metropolis,resfolder='./', update=args.update, loadweights=args.loadweights) 


    print sim.energy 
    sim.spins = sim.random_shift(sim.spins)
    print sim.energy 
