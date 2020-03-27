#!/bin/bash
#This script for running salt deletion

######################################
# Replace everything inside <...> with
# suitable settings for your jobs
# to force selection of PASCAL gpu's (faster!), set -l nodes=1:ppn=6:pascal 
######################################
# unused: PBS -q gpuq
# NOTE: reserve six cores as a way of blocking half of the node from other jobs.
#PBS -l nodes=1:ppn=6
#PBS -l walltime=12:00:00
#PBS -V
#PBS -j oe
#PBS -N __name__
#PBS -M kevinshen@ucsb.edu
#PBS -m abe
######################################
inputfile=params.in
outputfile=z.log
polyftsdir=~/lib/PolyFTS/bin/Release
scriptdir=/home/kshen/cg_chain/reference
N=__Nbead__
asmear=__asmear__
usmear=__usmear__
######################################

cd $PBS_O_WORKDIR
outdir=${PBS_O_WORKDIR}
rundir=${outdir}
username=`whoami`

############# TO USE LOCAL SCRATCH FOR INTERMEDIATE IO, UNCOMMENT THE FOLLOWING
#if [ ! -d /scratch_local/${username} ]; then
#  rundir=/scratch_local/${username}/${PBS_JOBID}
#  mkdir -p $rundir
#  cp ${PSB_O_WORKDIR}/* $rundir
#  cd $rundir
#fi
#####################################################

echo CUDA_VISIBLE:  $CUDA_VISIBLE_DEVICES
echo PBS_GPUFILE:   `cat $PBS_GPUFILE`
echo " "

# Fetch the device ID for the GPU that has been assigned to the job
GPUDEV=`cat $PBS_GPUFILE | awk '{print $1}'` #Takes $PBS_GPUFILE, cats it into a file, then (|) pipes it to an awk script '{...}' which prints the $1 variable of the line
if [ -z $GPUDEV ]; then
  echo "ERROR finding $PBS_GPUFILE; using default GPU deviceid=0"
  GPUDEV=0
fi

echo "Assigned GPU device: $GPUDEV"
echo " "
echo "=== === === Begin Running === === ==="
echo " "

source ~/.bashrc
conda activate py27
python --version
# Use default openmm test
python -m simtk.testInstallation

# Now test my script
#python simulateSetup.py --deviceid=$GPUDEV
#${scriptdir}/runSaltStagesI.sh $Nsalt $GPUDEV
cmd="python $scriptdir/run.py -np 1 -N $N -prodtime $[10*N*N] -eqtime $[N*N] -trajfreq $[N*N/100] -bond kg -initial zig"
echo $cmd
python $scriptdir/run.py -np 1 -N $N -prodtime $[10*N*N] -eqtime $[N*N] -trajfreq $[N*N/100] -bond kg -initial zig
python $scriptdir/run.py -np 1 -N $N -prodtime $[10*N*N] -eqtime $[N*N] -trajfreq $[N*N/100] -bond fjc -initial zig -asmear -usmear $usmear -dt 0.05


# Prepare the run by substituting the CUDA select device line
# Check whether the line exists first
#grep "CUDA_[Ss]elect[Dd]evice" ${inputfile} > /dev/null
#if [ $? -ne 0 ]; then
#  echo "CUDA_SelectDevice line not found in $inputfile"
#  exit 1
#fi
#sed -i "s/\(CUDA_[Ss]elect[Dd]evice\).*/\1 = ${GPUDEV}/g" ${inputfile}
#sed -i "s/\(CUDA_[Ss]elect[Dd]evice\).*/\1 = ${GPUDEV}/g" ${inputfile}

# Run the job
#${polyftsdir}/PolyFTSGPU.x ${inputfile} > ${outdir}/${outputfile}

# Copy back results
if [ "$rundir" != "$outdir" ]; then
  mv * ${outdir}
fi

# Force good exit code here - e.g., for job dependency
exit 0

