for T in 1.8 1.9 2.0 2.1 2.2 2.3 2.4 2.4 2.6 2.7 2.8 2.9 
do
  nohup python regression.py -f /Users/wanglei/Src/RBM/src/spins/ising_L16_T${T}_.dat -n_hidden 256 -n_epoch 100 -l & 
done
