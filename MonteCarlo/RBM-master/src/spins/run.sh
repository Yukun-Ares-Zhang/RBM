L=16 

workfolder=$PWD

resfolder=$PWD 
#for T in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
#for T in 1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9 2.0 
for T in 1.8 1.9 2.0 2.1 2.2 2.3 2.4 2.4 2.6 2.7 2.8 2.9 
#for T in 0.10 0.11 0.12 0.13 0.14 0.15 0.16 0.17 0.18 0.19 0.20
#for i in {1..200}
do 
    #T=$(python -c "import random; print '{:.5f}'.format(random.uniform(0., 1.0))")
    #T=`echo $i | awk '{printf "%.1f", $1/10}'`

    key=L${L}_T${T}_
    echo $key

    cd ${workfolder}
    sed -e '5s/.*/'T=${T}'/' params.in  > ${resfolder}/params_${key}
    ./ising < ${resfolder}/params_${key} >${resfolder}/ising_${key}.dat

    #cd ${resfolder}
    #grep -v Creat tmp${key} > configs${key}.dat 
    #rm tmp${key} 
done 
