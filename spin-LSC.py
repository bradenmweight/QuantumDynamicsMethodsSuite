import numpy as np
import multiprocessing as mp
import time, os, sys
import model_pyrazine as model
import numpy.linalg as LA
import subprocess as sp
import random

def getGlobalParams():
    global dtE, dtI, NSteps, NTraj, NStates, M, windowtype
    global adjustedgamma, NCPUS, initstate, dirName, method
    global fs_to_au, sampling, topDir, NSkip
    dtE = model.parameters.dtE
    dtI = model.parameters.dtI
    NSteps = model.parameters.NSteps
    NTraj = model.parameters.NTraj
    NStates = model.parameters.NStates
    M = model.parameters.M

    NCPUS = model.parameters.NCPUS
    initstate = model.parameters.initState
    dirName = model.parameters.dirName

    fs_to_au = 41.341 # a.u./fs
    NSkip = model.parameters.NSkip

def cleanDir(n):
    if ( os.path.exists( dirName + "/traj-"+str(n) ) ):
        sp.call("rm -r "+dirName+"/traj-"+str(n),shell=True)
    sp.call("mkdir -p "+dirName+"/traj-"+str(n),shell=True)

def initFiles(n):

    densityFile = open(dirName+"/traj-" + str(n) + "/density.dat","w")
    coherenceFile = open(dirName+"/traj-" + str(n) + "/coherence.dat","w")
    InitCondsFile = open(dirName+"/traj-" + str(n) + "/initconds.dat","w")
    RFile = open(dirName+"/traj-" + str(n) + "/RFile.dat","w")
    HelFile = open(dirName+"/traj-" + str(n) + "/Hel.dat","w")
    return densityFile, InitCondsFile, RFile, HelFile, coherenceFile

def closeFiles(densityFile, InitCondsFile, RFile, HelFile, coherenceFile):
    densityFile.close()    
    InitCondsFile.close()
    RFile.close()
    HelFile.close()
    coherenceFile.close()

def makeArrays():
    rho = np.zeros(( NStates,NStates ))
    return rho

def writeDensity(densityFile,coherenceFile,z,step,Hel):

    if ( step % NSkip == 0 ):
        rho = np.zeros((NStates,NStates),dtype=complex)
        rho = 0.5 * ( np.outer(np.conjugate(z), z) - gw * np.identity(NStates) ) 

        outArrayPOP = [ round ( step * dtI ,5), round ( step * dtI / fs_to_au,5 ) ]
        outArrayCOH = [ round ( step * dtI ,5), round ( step * dtI / fs_to_au,5 ) ]

        sumPOP = 0
        for n in range(NStates):
            outArrayPOP.append( np.real(rho[n,n]) )
            for m in range(n+1,NStates):
                outArrayCOH.append( np.real( rho[n,m] ) )
                outArrayCOH.append( np.imag( rho[n,m] ) )

        densityFile.write( "\t".join(map(str,outArrayPOP)) + "\n")
        coherenceFile.write( "\t".join(map(str,outArrayCOH)) + "\n")

        #print ( "\n\t(Time, POP) :", " ".join(map(str,outArrayPOP[1:])) )


def writeR(R,RFile):
    RFile.write(str(round(R[0],4)) + "\n")

def writeHel(Hel,HelFile):
    outList = []
    for i in range(NStates):
        for j in range(NStates):
            outList.append( round(Hel[i,j],6) )
    HelFile.write( "\t".join(map(str,outList))+"\n" )

def initMapping(InitCondsFile):# Initialization of the mapping Variables
    """
    Returns np.array zF and zB (complex)
    """
    global gw # Only depends on the number of states. So okay to be global

    Rw = 2*np.sqrt(NStates+1) # Radius of W Sphere
    gw = (2/NStates) * (np.sqrt(NStates + 1) - 1)

    # Z_mu = r_mu * Exp[i phi_mu] # Cartesian mapping variables: Z = X + i P
    # r_mu = np.sqrt( 2*( mu == lambda) + gw ) # Radius of mapping var when focused to lambda

    # Initialize mapping radii
    r = np.ones(( NStates )) * np.sqrt(gw)
    r[initstate] = np.sqrt( 2 + gw ) # Choose initial mapping state randomly

    z = np.zeros(( NStates ),dtype=complex)
    for i in range(NStates):
        phi = random.random() * 2 * np.pi # Azimuthal Angle -- Always Random
        z[i] = r[i] * ( np.cos( phi ) + 1j * np.sin( phi ) )
  
    rho = np.zeros((NStates,NStates),dtype=complex)
    rho = 0.5 * ( np.outer(np.conjugate(z),z) - gw * np.identity(NStates) )

    #print ("Initial Density Matrix:\n", np.round(rho,2) )

    return z

def propagateMapVars(z, VMat):
    """
    Updates mapping variables
    """
        
    Zreal = np.real(z)
    Zimag = np.imag(z)

    # Propagate Imaginary first by dt/2
    Zimag -= 0.5 * VMat @ Zreal * dtE

    # Propagate Real by full dt
    Zreal += VMat @ Zimag * dtE
    
    # Propagate Imaginary final by dt/2
    Zimag -= 0.5 * VMat @ Zreal * dtE

    return  Zreal + 1j*Zimag

def Force(dHel, R, z, dHel0):
    """
    F = F0 + Fm
    F0 = -GRAD V_0 (State-Independent)
    Fm = -GRAD V_m (State-Dependent and Traceless)
    V_m = 0.5 * SUM_(lam, u) <lam|V|u> z*_lam z'_u
    """

    action = np.real( ( np.outer(np.conjugate(z),z) - gw * np.identity(NStates) ) )

    F = np.zeros((len(R)))
    F -= dHel0
    for i in range(NStates):
        F -= 0.5 * dHel[i,i,:] * action[i,i]
        for j in range(i+1,NStates): # Double counting off-diagonal to save time
            F -= 2 * 0.5 * dHel[i,j,:] * action[i,j]
    return F

def VelVerF(R, P, z, RFile, HelFile): # Ionic position, ionic momentum, etc.
    
    v = P/M
    Hel = model.Hel(R) # Electronic Structure
    dHel = model.dHel(R)
    dHel0 = model.dHel0(R)
    EStep = int(dtI/dtE)
    
    for t in range( int(EStep/2) ): # Half-step Mapping
        z = propagateMapVars(z, Hel) * 1

    F1 = Force(dHel, R, z, dHel0)

    v += 0.5000 * F1 * dtI / M # Half-step velocity

    R += v * dtI # Full Step Position
    
    dHel = model.dHel(R)
    dHel0 = model.dHel0(R)
    F2 = Force(dHel, R, z, dHel0)
    
    v += 0.5000 * F2 * dtI / M # Half-step Velocity

    Hel = model.Hel(R) # Electronic Structure

    for t in range( int(EStep/2) ): # Half-step Mappings
        z = propagateMapVars(z, Hel) * 1

    return R, v*M, z, Hel

def RunIterations(n): # This is parallelized already. "Main" for each trajectory.

    print (f"Working in traj {n} for NSteps = {NSteps}")

    cleanDir(n)
    densityFile, InitCondsFile, RFile, HelFile, coherenceFile = initFiles(n) # Makes file objects
    rho = makeArrays()

    R,P = model.initR() # Initialize nuclear DOF
    z = initMapping(InitCondsFile) # Initialize electronic DOF

    Hel = model.Hel(R)
    dHij = model.dHel(R)
    for step in range(NSteps):
        #print ("Step:", step)
        Ugam = writeDensity(densityFile,coherenceFile,z,step,Hel)
        R, P, z, Hel = VelVerF(R, P, z, RFile, HelFile)
        
    
    closeFiles(densityFile, InitCondsFile, RFile, HelFile, coherenceFile)


### Start Main Program ###
if ( __name__ == "__main__"  ):

    getGlobalParams()
    start = time.time()

    print (f"There will be {NCPUS} cores with {NTraj} trajectories.")

    runList = np.arange(NTraj)
    with mp.Pool(processes=NCPUS) as pool:
        pool.map(RunIterations,runList)

    stop = time.time()
    print (f"Total Computation Time (Hours): {(stop - start) / 3600}")


