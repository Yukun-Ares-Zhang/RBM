#./collect_results.sh RBM acc 64 
for T in 1.8 1.9 2.0 2.1 2.2 2.3 2.4 2.4 2.6 2.7 2.8 2.9 
do
folder=/Users/wanglei/Src/RBM/data/nnfit/ising_L16_T${T}_nhidden100_

res=`grep -m 1 $2 $folder/$1.dat` 

mean=`echo $res | cut -d " " -f2` 
error=`echo $res | cut -d " " -f4` 
tau=`echo $res | cut -d " " -f5` 

echo $T $mean $error $tau 
done
