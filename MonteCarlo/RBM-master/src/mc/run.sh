for T in 1.8 1.9 2.0 2.1 2.2 2.3 2.4 2.4 2.6 2.7 2.8 2.9 
do
folder=/Users/wanglei/Src/RBM/data/nnfit/ising_L16_T${T}_nhidden256_
#nohup python mc.py -filename $folder/weights.hdf5 -update RBM -n_metropolis 1 -n_gibbs 1 > $folder/RBM.dat 2> $folder/error.dat & 
#nohup python mc.py -filename $folder/weights.hdf5 -update metropolis > $folder/Metropolis.dat  2> $folder/error.dat   & 
nohup python mc.py -filename $folder/weights.hdf5  -update RBM -n_metropolis 0 -n_gibbs 1 > $folder/RBM01.dat 2> $folder/error.dat & 
sleep 1
done
