#!/bin/bash
Ns=(10 30 100 300 1000 3000 10000)
Ns=(10 30 100 300 1000)
N=100
asmears=(0.3989422803 0.5 0.75 1.0 1.5 2.0 3.0 4.0 6.0 8.0 10.0 15.0 20.0)
usmears=(6.0383990 5.1215037 4.5999855 4.4879377 4.4324237 4.4191116 4.4123058 4.4106522 4.40980355 4.4095970 4.40952347 4.4094692 4.409455956)


#initname=init_gaussian.pdb
#prodtau=10000
#eqtau=1000

for ii in ${!asmears[@]}
do
    asmear=${asmears[$ii]}
    usmear=${usmears[$ii]}

    mydir=FJCN${N}_a${asmear}
    if [ ! -d $mydir ]
    then
        mkdir $mydir
    fi

    cd $mydir
    
    #run Kremer-Grest S.A.W.
    #python ../run.py -np 1 -N $N -prodtime $prodtau -eqtime $eqtau -bond kremer-grest -initial $initname
    
    #run smeared FJC
    #nohup python ../run.py -np 1 -N $N -prodtime $[10*N*N] -eqtime $[N*N] -trajfreq $[N*N/100] -bond kg -initial zig &
    cmd="python ../run.py -np 1 -N $N -prodtime $[25*N*N] -eqtime $[N*N] -trajfreq $[N*N/100] -bond fjc -initial zig -asmear $asmear -usmear $usmear -dt 0.05"
    #cmd="python ../run.py -np 1 -N $N -prodtime $[10*N*N] -eqtime $[N*N] -trajfreq $[N*N/100] -bond kg -initial zig"
    echo $cmd

    sed -e "s/__name__/$mydir/g" -e "s/__Nbead__/$N/g" -e "s/__asmear__/$asmear/g" -e "s/__usmear__/$usmear/g" ../zSubSm.sh > ./zSubSm.sh
    qsub zSubSm.sh

    cd ..
done

