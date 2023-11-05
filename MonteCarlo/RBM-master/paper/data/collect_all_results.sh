nh=100
for f in RBM01 RBM21 RBM41 Metropolis
do
for d in acc E
do
./collect_results.sh $f $d $nh > nhidden${nh}/${f}_${d}.dat 
done
done
