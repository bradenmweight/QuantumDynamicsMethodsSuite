import numpy as np
import multiprocessing as mp
import time, os, sys
import numpy.linalg as LA
import subprocess as sp
import random
import scipy as sc
from numba import jit

import model


def getGlobalParams():
    global dtE, dtI, NSteps, ESteps, NTraj, NStates, M, windowtype
    global adjustedgamma, NCPUS, initState, dirName, method
    global fs_to_au, sampling, topDir, NSkip, save_kernels, save_ABS
    dtE = model.parameters.dtE
    dtI = model.parameters.dtI
    ESteps = int(dtI/dtE)
    NSteps = model.parameters.NSteps
    NTraj = model.parameters.NTraj
    NStates = model.parameters.NStates
    M = model.parameters.M
    #sampling = model.parameters.sampling.lower()
    #windowtype = model.parameters.windowtype.lower()
    #adjustedgamma = model.parameters.adjustedgamma.lower()
    NCPUS = model.parameters.NCPUS
    initState = model.parameters.initState # Not needed but good for checking post-processing routine
    dirName = model.parameters.dirName  + "/Partial__" + sys.argv[1] + sys.argv[2]
    topDir = model.parameters.dirName
    #method = model.parameters.method.lower()
    fs_to_au = 41.341 # a.u./fs
    NSkip = model.parameters.NSkip
    try:
        save_kernels = model.parameters.save_kernels
    except AttributeError:
        save_kernels = False
    try:
        save_ABS = model.parameters.save_ABS
    except AttributeError:
        save_ABS = False
    

def cleanMainDir():
    #if ( os.path.exists(dirName) ):
    #    sp.call("rm -r "+topDir,shell=True)
    sp.call("mkdir -p "+dirName,shell=True)

def cleanDir(n):
    if ( os.path.exists( dirName + "/traj-"+str(n) ) ):
        sp.call("rm -r "+dirName+"/traj-"+str(n),shell=True)
    sp.call("mkdir -p "+dirName+"/traj-"+str(n),shell=True)

def initFiles(n):

    densityFile = open(dirName+"/traj-" + str(n) + "/population.dat","w")
    coherenceFile = open(dirName+"/traj-" + str(n) + "/coherence.dat","w")
    InitCondsFile = open(dirName+"/traj-" + str(n) + "/initconds.dat","w")
    RFile = open(dirName+"/traj-" + str(n) + "/RFile.dat","w")
    HelFile = open(dirName+"/traj-" + str(n) + "/Hel.dat","w")
    HadFile = open(dirName+"/traj-" + str(n) + "/Had.dat","w")
    mappingFile_F = open(dirName+"/traj-" + str(n) + "/mapping_F.dat","a")
    mappingFile_B = open(dirName+"/traj-" + str(n) + "/mapping_B.dat","a")
    gamma_mat_File = open(dirName+"/traj-" + str(n) + "/gamma_mat.dat","a")
    ABS_FILE = open(dirName+"/traj-" + str(n) + "/J_ABS.dat","a")
    return densityFile, InitCondsFile, RFile, HelFile, HadFile, coherenceFile, mappingFile_F, mappingFile_B, gamma_mat_File, ABS_FILE

def closeFiles(densityFile, InitCondsFile, RFile, HelFile, coherenceFile, mappingFile_F, mappingFile_B, ABS_FILE):
    densityFile.close()    
    InitCondsFile.close()
    RFile.close()
    HelFile.close()
    mappingFile_F.close()
    mappingFile_B.close()
    ABS_FILE.close()
    coherenceFile.close()

def makeArrays():
    hist = np.zeros(( NStates ))
    rho = np.zeros(( NStates,NStates ))
    Ugam = np.identity( (NStates), dtype=complex )
    return hist,rho,Ugam

@jit(nopython=True)
def update_Gamma( Ugam, Hel ):

    E,U = np.linalg.eigh(Hel)
    U   = U + 0.0j # For numba
    Udt = U @ np.diag( np.exp( -1j * E * dtI) ) @ U.T # Rotate eigenvalue exp matrix
    Ugam = Udt @ Ugam # Paper says U(t_N) * U(t_N-1) * U(t_N-2) * ... 
    
    return Ugam

@jit(nopython=True)
def build_kernels( z, z0, Ugam ):

    zF0, zF = z0[0], z[0] # Complex 1D array
    zB0, zB = z0[1], z[1] # Complex 1D array

    wF   = 0.5 * ( np.outer(zF[:], np.conjugate(zF0)[:] ) - gw * Ugam[:,:] )
    wB   = 0.5 * ( np.outer(zB[:], np.conjugate(zB0)[:] ) - gw * Ugam[:,:] )
    wB = np.conjugate( wB.T )

    return wB, wF

def writeDensity(densityFile,coherenceFile,i,wB,wF):

    outArrayPOP = [ round ( i * dtI ,5), round ( i * dtI / fs_to_au,5 ) ]
    outArrayCOH = [ round ( i * dtI ,5), round ( i * dtI / fs_to_au,5 ) ]

    POP = np.zeros(( NStates )) # Track population given initial state excitation
    A = np.zeros(( NStates, NStates ), dtype=complex) # Track population given initial state excitation
    A[initState,initState] = 1.0 + 0.0j
    for j in range( NStates ):
        B = np.zeros(( NStates, NStates ), dtype=complex)
        B[j,j] = 1.0 + 0.0j
        AwBw   = A @ wB @ B @ wF
        POP[j] = np.trace( np.real(AwBw) )

        outArrayPOP.append( POP[j] )

    densityFile.write( "\t".join(map("{:1.5f}".format,outArrayPOP)) + "\n")


def writeABS(ABS_FILE,z,i,z0,Ugam):

    outArrayABS = [ round ( i * dtI ,5), round ( i * dtI / fs_to_au,5 ) ]

    # Define dipole operator
    MU_p = np.zeros(( NStates, NStates ))
    MU_m = np.zeros(( NStates, NStates ))
    mu1 = -0.2
    mu2 = 1.0
    MU_m[0,1] = mu2
    MU_m[0,2] = mu1
    MU_m[1,3] = mu1
    MU_m[2,3] = mu2
    MU_p[1,0] = mu2
    MU_p[2,0] = mu1
    MU_p[3,1] = mu1
    MU_p[3,2] = mu2


    A = np.zeros(( NStates, NStates ), dtype=complex) # Track population given initial state excitation
    A[0,0] = 1.0 + 0.0j

    wB, wF = build_kernels( z, z0, Ugam )
    AwBw   = np.einsum( "ab,bc,cd,de,ef->af",  A, MU_m, wB, MU_p, wF ) # Which is correct ?
    J_ABS = np.einsum( "aa->",  AwBw )

    outArrayABS.append( J_ABS )

    ABS_FILE.write( "\t".join(map("{:1.5f}".format,outArrayABS)) + "\n")

def writeKernel( wB, wF, step, mappingFile_F, mappingFile_B, gamma_mat_File ):

    outF = [ round ( step * dtI ,5), round ( step * dtI / fs_to_au,5 ) ]
    outB = [ round ( step * dtI ,5), round ( step * dtI / fs_to_au,5 ) ]

    for j in range( NStates ):
        for k in range( NStates ):
            outF.append( wF[j,k] )
            outB.append( wB[j,k] )

    mappingFile_F.write( "\t".join(map("{:1.5f}".format,outF)) + "\n")
    mappingFile_B.write( "\t".join(map("{:1.5f}".format,outB)) + "\n")
    
    # TrAwBw[n,m,:,traj] = np.einsum( "...ii",  A @ wp @ B @ w ) # Will compute this matrix multiplication in post-processing


def writeR(R,RFile):
    RFile.write(str(round(R[0],4)) + "\n")

def writeHel(Hel,HelFile):
    outList = []
    for i in range(NStates):
        for j in range(NStates):
            outList.append( round(Hel[i,j],6) )
    HelFile.write( "\t".join(map(str,outList))+"\n" )

def writeHad(Hel,HadFile):
    Had, U = np.linalg.eigh(Hel)
    outList = []
    for i in range(NStates):
        outList.append( round(Had[i],6) )
    HadFile.write( "\t".join(map(str,outList))+"\n" )

def initMapping(InitCondsFile):# Initialization of the mapping Variables
    """
    Returns np.array zF and zB (complex)
    """
    global gw # Only depends on the number of states. So okay to be global

    Rw  = 2*np.sqrt(NStates+1) # Radius of W Sphere
    gw = (2/NStates) * (np.sqrt(NStates + 1) - 1) # ZPE of the W-sphere

    # Notes:
    # Z_mu = r_mu * Exp[i phi_mu] # Cartesian mapping variables: Z = X + i P
    # r_mu = np.sqrt( 2*( mu == lambda) + gw ) # Radius of mapping var when focused to lambda

    # Initialize mapping radii
    rF = np.ones(( NStates )) * np.sqrt(gw)
    initStateF = int(sys.argv[1])
    rF[initStateF] = np.sqrt( 2 + gw )

    rB = np.ones(( NStates )) * np.sqrt(gw)
    initStateB = int(sys.argv[2])
    rB[initStateB] = np.sqrt( 2 + gw )

    zF = np.zeros( (NStates), dtype=complex)
    zB = np.zeros( (NStates), dtype=complex)

    ### FOCUSED spin-PLDM INITIALIZATION ###
    for i in range(NStates):
        phiF = random.random() * 2 * np.pi # Azimuthal Angle -- Always Random
        zF[i] = rF[i] * ( np.cos( phiF ) + 1j * np.sin( phiF ) )
        phiB = random.random() * 2 * np.pi # Azimuthal Angle -- Always Random
        zB[i] = rB[i] * ( np.cos( phiB ) + 1j * np.sin( phiB ) )

    """
    # Test Initial Population
    A = np.zeros( (NStates,NStates), dtype=complex )
    A[initState,initState] = 1.0 + 0.0j
    wB, wF = build_kernels( np.array( [zF,zB] ), np.array( [zF,zB] ), np.identity(NStates) )
    print( np.real(np.trace( A @ wB @ A @ wB )) )
    exit()
    """

    return np.array( [zF,zB] ), np.array( [zF,zB] )

@jit(nopython=True)
def propagateMapVars(z, VMat):
    """
    Updates mapping variables
    """

    zF = z[0]
    zB = z[1]

    REzF = zF.real
    IMzF = zF.imag

    REzB = zB.real
    IMzB = zB.imag

    for t in range( ESteps ): # Full-step Electronic Mapping

        # Propagate Imaginary first by dt/2
        IMzF -= 0.5 * VMat @ REzF * dtE
        IMzB -= 0.5 * VMat @ REzB * dtE

        # Propagate Real by full dt
        REzF += VMat @ IMzF * dtE
        REzB += VMat @ IMzB * dtE
        
        # Propagate Imaginary final by dt/2
        IMzF -= 0.5 * VMat @ REzF * dtE
        IMzB -= 0.5 * VMat @ REzB * dtE

    zF = REzF + 1j*IMzF
    zB = REzB + 1j*IMzB
    z  = np.array([zF,zB])

    return z


#@jit(nopython=True)
def Force(dHel, R, z, z0, Ugam, wB, wF, dHel0):
    """
    F   = F0 + Fm
    F0  = -GRAD V_0 (State-Independent)
    Fm  = -GRAD V_m (State-Dependent and Traceless)
    V_m = 0.5 * SUM_(lam, u) <lam|V|u> z*_lam z'_u - gw * U[:,:]
    """

    action4 = 0.50 * np.real( wB + wF )

    F  = np.zeros( (len(R)) )
    F -= dHel0
    F -= np.einsum( "jkR,jk->R", dHel[:,:,:], action4[:,:] )
    return F

@jit(nopython=True)
def Force_TESTING(dHel, R, z, z0, Ugam, wB, wF, dHel0):
    """
    F = F0 + Fm
    F0 = -GRAD V_0 (State-Independent)
    Fm = -GRAD V_m (State-Dependent and Traceless)
    V_m = 0.5 * SUM_(lam, u) <lam|V|u> z*_lam z'_u - gw
    """

    zF0, zF = z0[0], z[0] # Complex 1D array
    zB0, zB = z0[1], z[1] # Complex 1D array

    # Should we include the time-dependent ZPE matrix here... ?
    # Should be z(t)*z(0) or z(t)*z(t) ?
    # IMPORTANT: action0 is the most stable. Tr[action0] = 1.0 for all times. All others do not.
    ##action0 = 0.25 * np.real( np.outer( zF, zF.conjugate()  )  + np.outer( zB.conjugate(),  zB )  - 2 * gw * np.identity(NStates) )
    ##action1 = 0.25 * np.real( np.outer( zF, zF0.conjugate() )  + np.outer( zB0.conjugate(), zB )  - 2 * gw * np.identity(NStates) )
    ##action2 = 0.25 * np.real( np.outer( zF, zF.conjugate()  )  + np.outer( zB.conjugate(),  zB )  - 2 * gw * Ugam )
    ##action3 = 0.25 * np.real( (np.outer(zF, zF0.conjugate()) - gw * Ugam.conjugate()) + np.conjugate(np.outer(zB, zB0.conjugate()) - gw * Ugam.conjugate()).T )
    action4 = 0.50 * np.real( wB + wF ) # Same as 3
    ##action5 = 0.25 * np.real( ( np.outer( zF, zF0.conjugate() )  + np.outer( zB0.conjugate(), zB ) ) )
    
    
    #print()
    #print( "0", np.sum(np.diagonal(action0)) )
    #print( "1", np.sum(np.diagonal(action1)) )
    #print( "2", np.sum(np.diagonal(action2)) )
    #print( "3", np.sum(np.diagonal(action3)) )
    #print( "4", np.sum(np.diagonal(action4)) )
    #print( "5", np.sum(np.diagonal(action5)) )
    #print( Ugam )

    F = np.zeros( (len(R)) )
    F -= dHel0
    for i in range(NStates):
        F -= dHel[i,i,:] * action4[i,i]
        for j in range(i+1,NStates): # Double counting off-diagonal to save time
            F -= 2 * dHel[i,j,:] * action4[i,j]
    return F

@jit(nopython=True)
def Force_noz0_noUgam(dHel, R, z, z0, Ugam, dHel0):
    """
    F = F0 + Fm
    F0 = -GRAD V_0 (State-Independent)
    Fm = -GRAD V_m (State-Dependent and Traceless)
    V_m = 0.5 * SUM_(lam, u) <lam|V|u> z*_lam z'_u - gw
    """

    zF = z[0]
    zB = z[1]

    # Should we include the time-dependent ZPE matrix here... ?
    # Should be z(t)*z(0) or z(t)*z(t) ?
    action = 0.5 * np.real( ( np.outer( zF.conjugate(), zF ) + np.outer( zB.conjugate(), zB ) - gw * np.identity(NStates) ) )

    F = np.zeros((len(R)))
    F -= dHel0
    for i in range(NStates):
        F -= 0.5 * dHel[i,i,:] * action[i,i]
        for j in range(i+1,NStates): # Double counting off-diagonal to save time
            F -= 2 * 0.5 * dHel[i,j,:] * action[i,j]
    return F

def do_Electronic_Structure( R ):
    Hel = model.Hel(R)
    dHel = model.dHel(R)
    dHel0 = model.dHel0(R)
    return Hel, dHel, dHel0

def VelVerF( R, P, z, z0, Ugam, wB, wF, F1, RFile, HelFile ): # Ionic position, ionic momentum, etc.
    
    R += P/M * dtI + 0.5000 * dtI * F1 / M # Full Step Nuclear Position
    
    Hel, dHel, dHel0 = do_Electronic_Structure( R )
    F2 = Force(dHel, R, z, z0, Ugam, wB, wF, dHel0)
    
    P += 0.5000 * dtI * (F1 + F2) # Half-step Nuclear Velocity

    Hel, dHel, dHel0 = do_Electronic_Structure( R )

    return R, P, z, Hel, F2

def RunIterations(n): # This is parallelized already. "Main" for each trajectory.

    print (f"Working in traj {n} for NSteps = {NSteps}")

    cleanDir(n)
    densityFile, InitCondsFile, RFile, HelFile, HadFile, coherenceFile, mappingFile_F, mappingFile_B, gamma_mat_File, ABS_FILE = initFiles(n) # Makes file objects
    hist,rho,Ugam = makeArrays()

    R,P = model.initR() # Initialize nuclear DOF

    z,z0 = initMapping(InitCondsFile)

    Hel, dHel, dHel0 = do_Electronic_Structure( R )
    F1 = Force(dHel, R, z, z0, Ugam, wB, wF, dHel0)
    for step in range(NSteps):
        #print ("Step:", step)
        wB, wF = build_kernels( z, z0, Ugam )
        if ( step % NSkip == 0 ):
            #writeHel(Hel,HelFile)
            #writeHad(Hel,HadFile)
            writeDensity(densityFile,coherenceFile,step,wB,wF)
            if ( save_kernels == True ): writeKernel( wB, wF, step, mappingFile_F, mappingFile_B, gamma_mat_File )
            if ( save_ABS == True ): writeABS(ABS_FILE,z,step,z0,Ugam)
        R, P, z, Hel, F1 = VelVerF(R, P, z, z0, Ugam, wB, wF, F1, RFile, HelFile)
        z = propagateMapVars(z, Hel)

        Ugam = update_Gamma( Ugam, Hel )
        
        
    
    closeFiles(densityFile, InitCondsFile, RFile, HelFile, coherenceFile, mappingFile_F, mappingFile_B, ABS_FILE)


### Start Main Program ###
if ( __name__ == "__main__"  ):

    getGlobalParams()
    cleanMainDir()
    start = time.time()

    print (f"There will be {NCPUS} cores with {NTraj} trajectories.")

    runList = np.arange(NTraj)
    with mp.Pool(processes=NCPUS) as pool:
        pool.map(RunIterations,runList)
    #RunIterations(0)

    stop = time.time()
    print (f"Total Computation Time (Hours): {(stop - start) / 3600}")


