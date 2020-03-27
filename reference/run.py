#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed March 25, 2020

@author: My Nguyen, Adapted by KS

Simulating Gaussian polymer solution in NVT/NPT ensemble
With or without the external potential

Flow of the script

Provide inputs in:
    0) define energy, length, mass scales
    1) dimensionless parameters defining system topology and interactions
       d) only if apply an sinusoidal external potential on the polymers, set Uext to 0 to disable this
    2) integration options
    3) set up reports

4)Convert dimensionless parameters to real units based on the scales provided in 0) since openmm only handle real units
5)-9) Create system and interactions
10)-12) Simulation
"""
#import mdtraj as md
from simtk import openmm, unit
#from simtk.unit import *
from simtk.openmm import app
import numpy as np
import simtk.openmm.app.topology as topology
import sys,argparse
parser = argparse.ArgumentParser(description='Simulation Properties')
parser.add_argument("-np", default=1, type=int, help="# polymer chains")
parser.add_argument("-ns", default=0, type=int, help="# solvent")
parser.add_argument("-N", default=2, type=int, help="degree of polymerization")
parser.add_argument("-bond", default='kg', type=str, choices=['kg','kremer','kremer-grest','g','h','gaussian','harmonic','fjc'], help="bond type")
parser.add_argument("-prodtime", default=1e5, type=int, help="production time (in tau)")
parser.add_argument("-eqtime", default=1e4, type=int, help="equilibration time (in tau)")
parser.add_argument("-trajfreq", default=1e2, type=int, help="dcd output frequency")
parser.add_argument("-L",default=10, type=float, help="boxL (irrelevant if not periodic)")
parser.add_argument("-initial", default=None, type=str, help="initial structure")
parser.add_argument("-verbose", action='store_true', help="verbose output")
parser.add_argument("-smear", action='store_true', help="whether to use smeared interactions")
parser.add_argument("-asmear", default=None, type=float, help="smearing length")
parser.add_argument("-usmear", default=0.0, type=float, help="smearing volume strength")
parser.add_argument("-bondl", default=1.0, type=float, help="FJC bond length")
parser.add_argument("-dt", default=0.01, type=float, help="reduced time step")
args = parser.parse_args()


#========================================
#0) DEFINE ENERGY, LENGTH & MAPS SCALES##
#========================================
epsilon  = 1 * unit.kilojoules_per_mole
sigma = 1 * unit.nanometers#unit.angstrom
tau = 1.0*unit.picoseconds
mass = tau**2/sigma**2*epsilon #12 * amu
N_av = unit.constants.AVOGADRO_CONSTANT_NA #6.022140857*10**23 /unit.mole
kB = unit.constants.BOLTZMANN_CONSTANT_kB  #1.380649*10**(-23)*joules/kelvin* N_av #joules/kelvin/mol
Tref = epsilon/kB/N_av

#====================================
#1) SYSTEM DIMENSIONLEPS PARAMETERS##
#   and hard-coded parameters
#====================================
#a)Molecules:
nspecies = 2
#number of polymers and degree of polymerization
n_p = args.np
DOP = args.N
#number of solvents
n_s = args.ns
Ntot = n_p*DOP + n_s

charge = 0.0 * unit.elementary_charge
#density in (length)**-3
reduced_density = 6000/(40**3)

#b)Bonded potential: k/2(r-l)**2
if args.bond.lower() in ['kg','kremer','kremer-grest']:
    bondparams={'type':'kremer-grest','k':30.,'R0':1.5}
elif args.bond.lower() in ['gaussian','harmonic','g','h']:
    #reduced equilibrium bond length
    #l = 1 
    #reduced force constant, note: k = 2*k_lammps 
    #k = 2 * 3000 
    bondparams={'type':'gaussian','k':2*100,'l':1}
elif args.bond.lower() in ['fjc']:
    bondparams={'type':'fjc','b':args.bondl}
else:
    raise(ValueError,'Unsupported bond type')


#c)Pair potential: WCA
#specified as [pp,ps,ss]
if args.asmear is None:
    pair_wca = True
    epslist = [1.0,1.0,1.0]
    siglist = [1.0,1.0,1.0]
    WCAeps = np.zeros( (nspecies,nspecies) )
    WCAsig = np.zeros( (nspecies,nspecies) )
    indices = np.triu_indices(nspecies,0)
    WCAeps[indices] = epslist
    WCAeps[indices[1],indices[0]] = epslist
    WCAsig[indices] = siglist
    WCAsig[indices[1],indices[0]] = siglist
    reduced_nonbondedCutoff = 2.0**(1./6.)
else:
    pair_wca = False
    pair_gaussian = True
    asmear = args.asmear
    usmear = args.usmear
    reduced_nonbondedCutoff = asmear*6.0

nonbondedMethod = openmm.CustomNonbondedForce.CutoffNonPeriodic
UseLongRangeCorrection = False

"""#Gaussian
u = -A exp(-Br^2)
B = 1
#interaction energy scale for each pair type
#polymer-polymer
A_PP = -100 
#solvent-solvent
A_PS = -100
#polymer-solvent
A_PS = -130
reduced_nonbondedCutoff = 10
nonbondedMethod = openmm.CustomNonbondedForce.CutoffPeriodic
#whether to add correction to the energy beyond nonbonded cutoff
UseLongRangeCorrection = False
"""


#d) External potential: applying sinusoidal external potential on polymer centroid, set Uext = 0 to disable 
extparams = {'mapping':6, 'Uext':0, 'Nperiod':1, 'axis':0, 'planeLoc':0, 'L':80}
"""
mapping = 6 #apply the external potential on the centroid of this many polymer monomers
Uext = 0 * mapping #amplitude of sinusoidal potential applied on the centroids
Nperiod = 1
axis  = 0
reduced_planeLoc = 0 
reduced_L = 80. #box length along axis where potential is applied
"""

#======================
#2) Integration Options
#======================
useNVT = True
useNPT = False
reduced_timestep = args.dt
reduced_temp = 1.
reduced_timedamp = 1. #time units
reduced_pressure = 1.
reduced_Pdamp = 1000*reduced_timestep #time units

steps = int(args.prodtime/reduced_timestep)#2e7
equilibrationSteps = int(args.eqtime/reduced_timestep)#1e7
#if platform is not set, openmm will try to select the fastest available Platform
platform = None
platformProperties = None
#platform = openmm.Platform.getPlatformByName('CPU')
#platformProperties = {'Precision': 'mixed'}
#platform = openmm.Platform.getPlatformByName('CUDA')

#=====================
#3) Reports
#=====================
trajfreq = int(args.trajfreq/reduced_timestep)
pdbfreq=trajfreq
logfreq=10000
dcdfreq=trajfreq
enforcePeriodic=False
#set up data and trajectory reports:
traj = 'trajectory.dcd'
pdbtraj = 'trajectory.pdb'
#if use mdtraj's reporter:
#dcdReporter = mdtraj.reporters.DCDReporter(traj, 5000)

#if don't have mdtraj, can use openmm's reporter to write out trajectory in pdb or dcd format
dcdReporter = app.dcdreporter.DCDReporter(traj, dcdfreq,enforcePeriodicBox=True)
pdbReporter = app.pdbreporter.PDBReporter(pdbtraj, pdbfreq)

dataReporter = app.statedatareporter.StateDataReporter('log.txt', logfreq, totalSteps=steps,
    step=True, speed=True, progress=True, potentialEnergy=True, kineticEnergy=True, 
    totalEnergy=True, temperature=True, volume=True, density=True, remainingTime=True,separator='\t')

dcdReporter_warmup = app.dcdreporter.DCDReporter('trajectory_warmup.dcd', dcdfreq,enforcePeriodicBox=False)
dataReporter_warmup= app.statedatareporter.StateDataReporter('log_warmup.txt', logfreq, totalSteps=equilibrationSteps,
    step=True, speed=True, progress=True, potentialEnergy=True, kineticEnergy=True,
    totalEnergy=True, temperature=True, volume=True, density=True, remainingTime=True,separator='\t')

dataReporter_stdout = app.statedatareporter.StateDataReporter(sys.stdout, 10*logfreq, totalSteps=steps+equilibrationSteps,
    step=True, speed=True, progress=True, potentialEnergy=True, kineticEnergy=True,
    totalEnergy=True, temperature=True, volume=True, density=True, remainingTime=True,separator='\t')

#=============================
#4) Converting to real units###
#=============================
nonbondedCutoff = reduced_nonbondedCutoff*sigma
planeLoc =  extparams['planeLoc']*sigma
L = extparams['L']*sigma
dt = reduced_timestep * tau
temperature = reduced_temp * epsilon/kB/N_av
friction = 1/(reduced_timedamp) / tau 
pressure = reduced_pressure * epsilon/N_av/sigma/sigma/sigma #epsilon/(sigma**3) * N_av**-1
barostatInterval = int(reduced_Pdamp/reduced_timestep)
print ('\n=== Parameters in real units ===')
print ('temperature:{}'.format(temperature))
print ('pressure:{}'.format(pressure))
print ('friction:{}'.format(friction))
print ('barostat interval:{}'.format(barostatInterval))
print ('nonbondedCutoff: {}'.format(nonbondedCutoff))

#========================================
#5) Create a system and add particles to it
#========================================
system = openmm.System()
print('\n=== Creating System ===')
# Particles are added one at a time
# Their indices in the System will correspond with their indices in the Force objects we will add later
print ("Adding {} polymer atoms into system".format(n_p*DOP))
for index in range(n_p*DOP):
    system.addParticle(mass)
print ("Adding {} solvent atoms into system".format(n_s))
for index in range(n_s):
    system.addParticle(mass)
print("Total number of paricles in system: {}".format(system.getNumParticles()))
# Set the periodic box vectors:
box_edge = [args.L*sigma] * 3
"""
number_density = reduced_density / sigma**3
volume = Ntot * (number_density ** -1)
if Uext == 0: #use cubic box if no external potential is applied
    box_edge = [volume ** (1. / 3.)] * 3
else: #use rectangular box where L is the length of box in the direction of the external potential
    box_edge_short = (volume/L)**(1./2.)
    box_edge = [box_edge_short]*3
    box_edge[axis] = L
"""
print ('Box dimensions {}'.format(box_edge))
box_vectors = np.diag([edge/sigma for edge in box_edge]) * sigma
system.setDefaultPeriodicBoxVectors(*box_vectors)

#==================
#6) Create topology
#==================
#Topology consists of a set of Chains 
#Each Chain contains a set of Residues, 
#and each Residue contains a set of Atoms.
nmols = [n_p, n_s]
residues = [["Poly"], ["Sol"]]
PolymerAtomList = DOP*['P']
atomList = {"Poly":PolymerAtomList,"Sol":['S']} #list of atoms in each residue
elements = {"P":app.element.Element(200, 'Polymer', 'gP', mass),
            "S":app.element.Element(201, 'Solvent', 'gS', mass)}
def makeTop(n_p,n_s,DOP):
    nmols = [n_p, n_s]
    top = topology.Topology()
    for ispec in range(len(nmols)): #loop over each species
        for imol in range(nmols[ispec]): #loop over each molecule within species
            chain = top.addChain() #create chain for each molecule
            for ires in range( len(residues[ispec]) ): #polymer and solvent each has one residue
                resname = residues[ispec][ires]
                res = top.addResidue( resname, chain)
                atoms = atomList[resname]
                for atomInd,atomName in enumerate(atoms):
                    el = elements[atomName]
                    if atomInd > 0:
                        previousAtom = atom
                    atom = top.addAtom( atomName, el, res )
                    if atomInd > 0:
                        top.addBond(previousAtom,atom)
                        if bondparams['type'] == 'fjc':
                            system.addConstraint( previousAtom.index, atom.index, bondparams['b']*sigma )
    return top
print ("\n=== Creating topology ===")
top = makeTop(n_p,n_s,DOP)

#===========================
#7) create BondForce
#===========================
if bondparams['type'] == 'gaussian':
    print('... adding gaussian bond')
    bondedForce = openmm.HarmonicBondForce()
    atomIndThusFar = 0 
    for imol in range(n_p): #loop over all polymer chain
        counter = 0
        for atomInd in range(atomIndThusFar,atomIndThusFar+DOP-1):
            bondedForce.addBond(atomInd,atomInd+1,bondparams['l']*sigma,bondparams['k']*epsilon/(sigma*sigma))
            counter += 1
        atomIndThusFar += counter + 1 #skip the last atom in polymer chain
    system.addForce(bondedForce)
elif bondparams['type'] == 'kremer-grest':
    print('... adding FENE')
    energy_function = '- 0.5 * kbond*R0^2 * log( 1.0 - (r/R0)^2 )'
    bondedForce = openmm.CustomBondForce( energy_function )
    bondedForce.addPerBondParameter('kbond')
    bondedForce.addPerBondParameter('R0')
    #bondedForce.addGlobalParameter('kbond', (bondparams['k']*epsilon/sigma/sigma).value_in_unit(unit.kilojoules_per_mole))
    atomIndThusFar = 0 
    for imol in range(n_p): #loop over all polymer chain
        counter = 0
        for atomInd in range(atomIndThusFar,atomIndThusFar+DOP-1):
            bondedForce.addBond(atomInd,atomInd+1, [bondparams['k']*epsilon/sigma/sigma, bondparams['R0']*sigma])
            counter += 1
        atomIndThusFar += counter + 1 #skip the last atom in polymer chain
    system.addForce(bondedForce)
    #bondedForce.addGlobalParameter('R0', (bondparams['R0']*sigma).value_in_unit(unit.nanometers))
elif bondparams['type'] == 'fjc':
    print("... constrained bond length for FJC")
    


#================================
#8) create custom nonbonded force
#================================
print('\n=== Creating Nonbonded Force===')
polymer_set = set()
solvent_set = set()
for atom in top.atoms():
    if atom.residue.name in ['Poly']:
        polymer_set.add(atom.index)
    else:
        solvent_set.add(atom.index)
all_atoms = polymer_set.union(solvent_set)

if pair_wca:
    #PP
    energy_function =  '4*epsPP*( sigr6^2 - sigr6 + 0.25);'
    energy_function += 'sigr6 = (sigPP/r)^6;'
    PP_nonbondedForce = openmm.CustomNonbondedForce(energy_function)
    PP_nonbondedForce.setNonbondedMethod(nonbondedMethod)
    PP_nonbondedForce.addGlobalParameter('epsPP', WCAeps[0,0]*epsilon )
    PP_nonbondedForce.addGlobalParameter('sigPP', WCAsig[0,0]*sigma ) 
    PP_nonbondedForce.addInteractionGroup(polymer_set,polymer_set)
    for i in range(system.getNumParticles()):
        PP_nonbondedForce.addParticle()
    PP_nonbondedForce.setCutoffDistance(nonbondedCutoff)
    PP_nonbondedForce.setUseLongRangeCorrection(UseLongRangeCorrection)
    system.addForce(PP_nonbondedForce)
    
    #PS
    energy_function =  '4*epsSS*( sigr6^2 - sigr6 + 0.25);'
    energy_function += 'sigr6 = (sigSS/r)^6;'
    PS_nonbondedForce = openmm.CustomNonbondedForce(energy_function)
    PS_nonbondedForce.setNonbondedMethod(nonbondedMethod)
    PS_nonbondedForce.addGlobalParameter('epsSS', WCAeps[1,1]*epsilon )
    PS_nonbondedForce.addGlobalParameter('sigSS', WCAsig[1,1]*sigma ) 
    PS_nonbondedForce.addInteractionGroup(solvent_set,solvent_set)
    for i in range(system.getNumParticles()):
        PS_nonbondedForce.addParticle()
    PS_nonbondedForce.setCutoffDistance(nonbondedCutoff)
    PS_nonbondedForce.setUseLongRangeCorrection(UseLongRangeCorrection)
    system.addForce(PS_nonbondedForce)
 

    #PS
    energy_function =  '4*epsPS*( sigr6^2 - sigr6 + 0.25);'
    energy_function += 'sigr6 = (sigPS/r)^6;'
    PS_nonbondedForce = openmm.CustomNonbondedForce(energy_function)
    PS_nonbondedForce.setNonbondedMethod(nonbondedMethod)
    PS_nonbondedForce.addGlobalParameter('epsPS', WCAeps[0,1]*epsilon )
    PS_nonbondedForce.addGlobalParameter('sigPS', WCAsig[0,1]*sigma ) 
    PS_nonbondedForce.addInteractionGroup(polymer_set,solvent_set)
    for i in range(system.getNumParticles()):
        PS_nonbondedForce.addParticle()
    PS_nonbondedForce.setCutoffDistance(nonbondedCutoff)
    PS_nonbondedForce.setUseLongRangeCorrection(UseLongRangeCorrection)
    system.addForce(PS_nonbondedForce)
 
elif pair_gaussian:
    #Polymer-polymer
    bPP = 1./4./asmear**2.0
    aPP = usmear/(np.pi/bPP)**1.5
    PP_nonbondedForce = openmm.CustomNonbondedForce('aPP*(exp(-bPP*r^2) - exp(-bPP*rcutPP^2))')
    PP_nonbondedForce.setNonbondedMethod(nonbondedMethod)
    PP_nonbondedForce.addGlobalParameter('bPP',bPP/(sigma*sigma)) #length^-2
    PP_nonbondedForce.addGlobalParameter('aPP',aPP*epsilon) #energy/mol
    PP_nonbondedForce.addGlobalParameter('rcutPP',nonbondedCutoff) #sigma
    PP_nonbondedForce.addInteractionGroup(polymer_set,polymer_set)
    for i in range(system.getNumParticles()):
        PP_nonbondedForce.addParticle()
    PP_nonbondedForce.setCutoffDistance(nonbondedCutoff)
    PP_nonbondedForce.setUseLongRangeCorrection(UseLongRangeCorrection)
    system.addForce(PP_nonbondedForce)
    
else:
    #gaussianFunc = '-A*exp(-B*r^2)'
    #for each pair interaction type, need to add ALL atoms in the system to the force object, only paricles in the InteractionGroup will interact with each other
    Polymer = set()
    Solvent = set()
    for atom in top.atoms():
        if atom.residue.name in ['Poly']:
            Polymer.add(atom.index)
        else:
            Solvent.add(atom.index)
    all_atoms = Polymer.union(Solvent)

    #Polymer-polymer and Solvent-Solvent:
    PP_nonbondedForce = openmm.CustomNonbondedForce('-APP*exp(-B*r^2)')
    PP_nonbondedForce.setNonbondedMethod(nonbondedMethod)
    PP_nonbondedForce.addGlobalParameter('B',B/(sigma*sigma)) #length^-2
    PP_nonbondedForce.addGlobalParameter('APP',A_PP*epsilon) #energy/mol
    PP_nonbondedForce.addInteractionGroup(Polymer,Polymer)
    for i in range(system.getNumParticles()):
        PP_nonbondedForce.addParticle()
    PP_nonbondedForce.setCutoffDistance(nonbondedCutoff)
    PP_nonbondedForce.setUseLongRangeCorrection(UseLongRangeCorrection)
    system.addForce(PP_nonbondedForce)
    print ("\nNumber of particles in PP nonbonded force:{}".format(PP_nonbondedForce.getNumParticles()))
    #Solvent-solvent:
    PS_nonbondedForce = openmm.CustomNonbondedForce('-ASS*exp(-B*r^2)')
    PS_nonbondedForce.setNonbondedMethod(nonbondedMethod)
    PS_nonbondedForce.addGlobalParameter('B',B/(sigma*sigma)) #length^-2
    PS_nonbondedForce.addGlobalParameter('ASS',A_SS*epsilon) #energy/mol
    PS_nonbondedForce.addInteractionGroup(Solvent,Solvent)
    for i in range(system.getNumParticles()):
            PS_nonbondedForce.addParticle()
    PS_nonbondedForce.setCutoffDistance(nonbondedCutoff)
    PS_nonbondedForce.setUseLongRangeCorrection(UseLongRangeCorrection)
    system.addForce(PS_nonbondedForce)
    print ("Number of particles in PS nonbonded force:{}".format(SS_nonbondedForce.getNumParticles()))

    #Polymer-solvent:
    PS_nonbondedForce = openmm.CustomNonbondedForce('-APS*exp(-B*r^2)')
    PS_nonbondedForce.setNonbondedMethod(nonbondedMethod)
    PS_nonbondedForce.addGlobalParameter('B',B/(sigma*sigma)) #length^-2
    PS_nonbondedForce.addGlobalParameter('APS',A_PS*epsilon) #energy/mol
    PS_nonbondedForce.addInteractionGroup(Polymer,Solvent)
    for i in range(system.getNumParticles()):
            PS_nonbondedForce.addParticle()
    PS_nonbondedForce.setCutoffDistance(nonbondedCutoff)
    PS_nonbondedForce.setUseLongRangeCorrection(UseLongRangeCorrection)
    system.addForce(PS_nonbondedForce)
    print ("Number of particles in PS nonbonded force:{}".format(PS_nonbondedForce.getNumParticles()))

    #force.setUseSwitchingFunction(True) # use a smooth switching function to avoid force discontinuities at cutoff
    #force.setSwitchingDistance(0.8*nonbondedCutoff)

#=============================
#9) create external potential
#=============================
external={"U":extparams['Uext']*epsilon,"NPeriod":extparams['Nperiod'],"axis":extparams['axis'] ,"planeLoc":extparams['planeLoc']}
direction=['x','y','z']
ax = external["axis"]
#atomsInExtField = [elementMap[atomname]]
if external["U"] > 0.0 * unit.kilojoules_per_mole:
        print('\n=== Creating sinusoidal external potential in the {} direction'.format(direction[axis]))
        energy_function = 'U*sin(2*pi*NPeriod*({axis}1-r0)/L)'.format(axis=direction[ax])
        fExt = openmm.CustomCentroidBondForce(1,energy_function)
        fExt.addGlobalParameter("U", external["U"])
        fExt.addGlobalParameter("NPeriod", external["NPeriod"])
        fExt.addGlobalParameter("pi",np.pi)
        fExt.addGlobalParameter("r0",external["planeLoc"])
        fExt.addGlobalParameter("L",box_edge[ax])
        atomThusFar = 0
        for i in range(int(n_p*DOP/mapping)): #looping through CG beads
            fExt.addGroup(range(atomThusFar,atomThusFar+mapping)) #assuming the first n_p*DOP atoms are polymer atoms and atom index increases along chain
            fExt.addBond([i], [])
            atomThusFar += mapping
        system.addForce(fExt)

#==============================
#10) Prepare the Simulation ##
#==============================
if useNPT:
    system.addForce(openmm.MonteCarloBarostat(pressure, temperature, barostatInterval))
integrator = openmm.LangevinIntegrator(temperature, friction, dt)
#use platform if specified, otherwise let omm decide
if platform:
    if platformProperties:
        simulation = app.Simulation(top,system, integrator, platform,platformProperties)
    else:
        simulation = app.Simulation(top,system, integrator, platform)
else:
    simulation = app.Simulation(top, system, integrator)

#initialize positions
if args.initial is None:
    positions = [box_edge[0]/sigma,box_edge[1]/sigma,box_edge[2]/sigma]*sigma * np.random.rand(Ntot,3)
elif args.initial.lower() in ['test','testpair']:
    positions = sigma * np.array( [[0,0,0],[0,0,1.0]] )
elif args.initial.lower() in ['rw','randomwalk','random_walk','random-walk']:
    raise(ValueError,'random walk not imlemented yet')
    positions = np.zeros( (Ntot,3) )
    bondstep = 1.0
    prevbond = 0
    bonddirs = [0,1,2,3,4,5]
    prevpos = np.array([0.,0.,0.])
    bondunit = [0.,0.,1.]
    for ii in range(1, Ntot):
        positions[ii,:] = prevpos + bondunit
elif args.initial.lower() in ['zig']:
    expectedRee = Ntot**0.588
    num_segments = int(np.ceil( expectedRee / 3. ))
    mon_per_segment = int(np.ceil( Ntot/num_segments ))

    positions = np.zeros( (Ntot,3) )
    prev_pos = np.array([0.,0.,0.])
    num_mon_added = 1
    for ii in range(0,num_segments):
        for jj in range(mon_per_segment):
            if jj < mon_per_segment-2:
                direction = np.array([0.,0.,1])*(-1)**(ii)
            else:
                if np.random.rand(1) < 0.5:
                    direction = np.array([0.,1.,0.])    
                else:
                    direction = np.array([1.,0.,0.])    

            if num_mon_added < Ntot:
                next_pos = prev_pos + direction
                positions[num_mon_added,:] = next_pos
                prev_pos = next_pos
                if args.verbose: print('added {}th monomer at {}'.format(num_mon_added, next_pos))
                num_mon_added += 1
            else:
                print("finished building zig zag chain")
                break
else:
    import mdtraj
    positions = mdtraj.load(args.initial).openmm_positions(0) 
    #positions = app.PDBFile(args.initial).positions

simulation.context.setPositions(positions)
initialpdb = 'trajectory_init.pdb'
app.PDBFile.writeModel(simulation.topology, positions, open(initialpdb,'w'))
state = simulation.context.getState(getEnergy=True,getPositions=True,enforcePeriodicBox=enforcePeriodic)
if args.verbose: print(state.getPositions())
print('initial energy: {}'.format(state.getPotentialEnergy()))
#write initial positions to pdb

#Restart and Check point:
#to load a saved state: simulation.loadState('output.xml') or simulation.loadCheckpoint('checkpnt.chk')
simulation.reporters.append(app.checkpointreporter.CheckpointReporter('checkpnt.chk', 1000*logfreq))
simulation.reporters.append(dataReporter_stdout)

#============================
#11) Minimize and Equilibrate
#============================
if useNVT:
    ensemble = "NVT"
else:
    ensemble = "NPT"

print ("\n=== Initialize and run {} simulation ===".format(ensemble))
if bondparams['type'] == 'fjc':
    print('...FJC, applying constraints first')
    simulation.context.applyConstraints(1e-5)

print ("Running energy minimization")
simulation.minimizeEnergy(maxIterations=1000)
#simulation.context.setVelocitiesToTemperature(temperature*3)
state = simulation.context.getState(getEnergy=True,getPositions=True,enforcePeriodicBox=enforcePeriodic)
if args.verbose: print(state.getPositions())
app.PDBFile.writeModel(simulation.topology, state.getPositions(), open('minimized.pdb','w'))
print('minimized energy: {}'.format(state.getPotentialEnergy()))

print ("\nRunning equilibration for {} steps ({} tau), log every {} steps ({} tau)".format(equilibrationSteps,equilibrationSteps*reduced_timestep, dcdfreq,dcdfreq*reduced_timestep))
simulation.reporters.append(dataReporter_warmup)
simulation.reporters.append(dcdReporter_warmup)
simulation.step(equilibrationSteps)
simulation.saveState('output_warmup.xml')
state = simulation.context.getState(getEnergy=True,getPositions=True,enforcePeriodicBox=enforcePeriodic)
app.PDBFile.writeModel(simulation.topology, state.getPositions(), open('equilibrated.pdb','w'))

#=============
#12) Simulate
#=============
print ("\nRunning production for {} steps ({} tau), log every {} steps ({} tau)".format(steps,steps*reduced_timestep, dcdfreq,dcdfreq*reduced_timestep))
simulation.reporters.append(dcdReporter)
simulation.reporters.append(pdbReporter)
simulation.reporters.append(dataReporter)
simulation.currentStep = 0
simulation.step(steps)
simulation.saveState('output.xml')

state = simulation.context.getState(getEnergy=True,getPositions=True,enforcePeriodicBox=enforcePeriodic)
app.PDBFile.writeModel(simulation.topology, state.getPositions(), open('final.pdb','w'))

#t = md.load(traj,top=initialpdb)
#trajOut = traj.split('.')[0] + '.pdb'
#t.save(trajOut)
#trajOut = traj.split('.')[0] + '.lammpstrj'
#t.save(trajOut)

