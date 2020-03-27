#!/bin/bash
Ns=(10 30 100 300 1000 3000 10000)
Ns=(10 30 100 300 1000)

initname=init_gaussian.pdb
prodtau=10000
eqtau=1000

for N in ${Ns[@]}
do
    mydir=N$N
    if [ ! -d $mydir ]
    then
        mkdir $mydir
    fi

    cd $mydir
    
    #use Gaussian chain to initialize chain structure
    #python ../run.py -np 1 -N $N -prodtime 10 -eqtime 10 -bond gaussian
    #mv equilibrated.pdb $initname

    #run Kremer-Grest S.A.W.
    #python ../run.py -np 1 -N $N -prodtime $prodtau -eqtime $eqtau -bond kremer-grest -initial $initname
    
    echo $[N*N/10]

    #nohup python ../run.py -np 1 -N $N -prodtime $[10*N*N] -eqtime $[N*N] -trajfreq $[N*N/100] -bond kg -initial zig &
    cmd="python ../run.py -np 1 -N $N -prodtime $[10*N*N] -eqtime $[N*N] -trajfreq $[N*N/100] -bond kg -initial zig"
    echo $cmd

    sed -e "s/__name__/$mydir/g" -e "s/__Nbead__/$N/g" ../zSub.sh > ./zSub.sh

    cd ..
done

